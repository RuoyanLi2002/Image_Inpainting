import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from numba import njit

from pyjuice.nodes import InputNodes
from pyjuice.nodes import ProdNodes


def grey_index_to_rgb_rc(index, args):
    h = args.data_shape[1]
    w = args.data_shape[2]
    row = index // w
    col = index % w

    return row, col

def grey_index_to_rgb_index(index, args):
    h = args.data_shape[1]
    w = args.data_shape[2]
    row, col = grey_index_to_rgb_rc(index, args)

    red_index = row * w + col
    green_index = h * w + row * w + col
    blue_index = 2 * h * w + row * w + col
    return red_index, green_index, blue_index

def correct_prod_node(scope, ls, args):
    if len(scope) == args.split_intervals[0] * args.split_intervals[1] * args.split_intervals[2]:
        set_scope = set(scope)
        set_ls = set(ls)

        return set_ls.issubset(set_scope)
        
    return False

def rgb_index_to_prod_node_index(ns, red_index, green_index, blue_index, args):
    for temp_ns in ns:
        if isinstance(temp_ns, ProdNodes) and correct_prod_node(temp_ns.scope.to_list(), [red_index, green_index, blue_index], args):
            s, e = temp_ns._output_ind_range

            return s, e

def input_node_se(ns, index):
    for temp_ns in ns:
        if isinstance(temp_ns, InputNodes) and temp_ns.scope.to_list() == [index]:
            s, e = temp_ns._output_ind_range

            return s, e

def create_circuit_info(ns, pc, args):
    file_path = os.path.join(args.cache_dir, "circuit.pkl")
    if os.path.exists(file_path):
        print(f"Loading circuit data from {file_path}")
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        param_array = data["param_array"]
        grey_idx_to_prod_se_array = data["grey_idx_to_prod_se_array"]
        color_idx_to_input_se_array = data["color_idx_to_input_se_array"]

    else:
        param_dict = {}
        with torch.no_grad():
            for idx, input_layer in enumerate(pc.input_layer_group):
                for node in input_layer.nodes:
                    index = node.scope.to_list() # [0]
                    params = node._params # [16384]
                    params = params.view(-1, 256) # [64, 256] (All row sum to 1 CHECKED)
                    param_dict[index[0]] = params

        total_input_nodes = len(param_dict)
        param_array = np.zeros((total_input_nodes, 64, 256))
        
        for i, (key, value) in enumerate(param_dict.items()):
            param_array[int(key), :, :] = value.numpy()

        grey_idx_to_prod_se_dict = {}
        color_idx_to_input_se_dict = {}
        for grey_index in range(args.data_shape[1]*args.data_shape[2]):
            red_index, green_index, blue_index = grey_index_to_rgb_index(grey_index, args)
            grey_start, grey_end = rgb_index_to_prod_node_index(ns, red_index, green_index, blue_index, args)
            grey_idx_to_prod_se_dict[grey_index] = [grey_start, grey_end]
            
            red_s, red_e = input_node_se(ns, red_index)
            green_s, green_e = input_node_se(ns, green_index)
            blue_s, blue_e = input_node_se(ns, blue_index)
            color_idx_to_input_se_dict[red_index] = [red_s, red_e]
            color_idx_to_input_se_dict[green_index] = [green_s, green_e]
            color_idx_to_input_se_dict[blue_index] = [blue_s, blue_e]

        grey_idx_to_prod_se_array = np.array(list(grey_idx_to_prod_se_dict.values()))
        grey_idx_to_prod_se_array = grey_idx_to_prod_se_array.reshape(256, 2)

        sorted_keys = sorted(color_idx_to_input_se_dict.keys())
        reordered_values = [color_idx_to_input_se_dict[key] for key in sorted_keys]
        color_idx_to_input_se_array = np.concatenate(reordered_values)
        color_idx_to_input_se_array = color_idx_to_input_se_array.reshape(-1, 2)

        data = {"param_array": param_array, "grey_idx_to_prod_se_array": grey_idx_to_prod_se_array, "color_idx_to_input_se_array": color_idx_to_input_se_array}
        with open(file_path, 'wb') as file:
                pickle.dump(data, file)
        
        print(f"Circuit data saved at {file_path}")

    print(f"param_array: {param_array.shape}") # (768, 64, 256)
    print(f"grey_idx_to_prod_se_array: {grey_idx_to_prod_se_array.shape}") # (256, 2)
    print(f"color_idx_to_input_se_array: {color_idx_to_input_se_array.shape}") # (768, 2)

    return param_array, grey_idx_to_prod_se_array, color_idx_to_input_se_array

def create_grey_info(args):
    file_path = os.path.join(args.cache_dir, "grey.pkl")
    if os.path.exists(file_path):
        print(f"Loading grey data from {file_path}")
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        arr_valid_ycocg = data["arr_valid_ycocg"]
        arr_num_valid_ycocg = data["arr_num_valid_ycocg"]

    else:
        ls_valid_ycocg, ls_valid_ycocg_tensor, ls_num_valid_ycocg = find_all_valid_ycocg()
        
        max_num_elements = max(len(sub_list[0]) for sub_list in ls_valid_ycocg)
        
        arr_valid_ycocg = np.zeros((256, 3, max_num_elements))
        arr_num_valid_ycocg = np.zeros(256)

        for i, sub_list in enumerate(ls_valid_ycocg):
            current_len = len(sub_list[0])
            arr_num_valid_ycocg[i] = current_len
            for j in range(3):
                arr_valid_ycocg[i, j, :current_len] = sub_list[j]

        data = {"arr_valid_ycocg": arr_valid_ycocg, "arr_num_valid_ycocg": arr_num_valid_ycocg}
        with open(file_path, 'wb') as file:
                pickle.dump(data, file)
        
        print(f"Grey data saved at {file_path}")

    print(f"arr_valid_ycocg: {arr_valid_ycocg.shape}") # (256, 3, 32767)
    print(f"arr_num_valid_ycocg: {arr_num_valid_ycocg.shape}") # (256,)

    return arr_valid_ycocg, arr_num_valid_ycocg

def compute_grey_prob(args, pc_num_elements, grey_arr, param_arr, grey_idx_to_prod_se_arr, arr_valid_ycocg, arr_num_valid_ycocg):
    grey_prob = np.full((pc_num_elements, 1), -np.inf, dtype = np.float32)
    print(f"grey_arr: {grey_arr.shape}")
    
    for grey_index, grey in enumerate(grey_arr):
        grey_start, grey_end = grey_idx_to_prod_se_arr[grey_index]
        
        h = args.data_shape[1]
        w = args.data_shape[2]
        row = grey_index // w
        col = grey_index % w

        red_index = row * w + col

        red_params = param_arr[red_index] # (64, 256)
        red_vals = red_params[:, grey] # (64, 1)
        
        grey_vals = np.log(red_vals)
        grey_vals = grey_vals.reshape(-1, 1)

        if not is_left(grey_index):
            grey_vals = np.zeros_like(grey_vals)

        if np.all(grey_prob[grey_start : grey_end, :] == -np.inf):
            grey_prob[grey_start : grey_end, :] = grey_vals
        else:
            grey_prob[grey_start : grey_end, :] += grey_vals

    return grey_prob # (pc_num_elements, 1)

def is_left(index):
    # Calculate the column position
    column = index % 16
    
    # Determine if it's on the left or right side
    if column <= 7:
        return True
    else:
        return False