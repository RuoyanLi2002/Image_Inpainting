import time
import math
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import pyjuice as juice
from functools import partial

from forward_backward_algo import forward, backward
from utils import grey_index_to_rgb_index, grey_index_to_rgb_rc

from matplotlib import pyplot as plt

# def sample_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color, groun_truth_color_pixel):
#     red_index, green_index, blue_index = grey_index_to_rgb_index(grey_index, args)
    
#     param_array = pc_args["param_array"]
#     if color == "red":
#         color_val = param_array[red_index, :, :]
#     elif color == "green":
#         color_val = param_array[green_index, :, :]
#     else:
#         color_val = param_array[blue_index, :, :] # (64, 256)
#     color_val = torch.from_numpy(color_val)

#     grey_start, grey_end = pc_args["grey_idx_to_prod_se_array"][grey_index]
#     aaa = pc_args["grey_idx_to_prod_se_array"][grey_index]

#     flow_val = flow[grey_start : grey_end, ]
#     flow_val = flow_val.unsqueeze(-1) # [64, 1]
 
#     result = flow_val * color_val[:, candidate_color]
#     column_sums = result.sum(dim=0)
#     column_sums = column_sums /torch.sum(column_sums)
    
#     # print(candidate_color.device)
#     # print(groun_truth_color_pixel.device)
#     max_column_index = candidate_color == groun_truth_color_pixel
    
#     return candidate_color[max_column_index], torch.log(column_sums[max_column_index])

def sample_pixel(pc, ns, args, pc_args, flow, grey_index):
    red_index, green_index, blue_index = grey_index_to_rgb_index(grey_index, args)

    param_array = pc_args["param_array"]
    color_val = param_array[red_index, :, :]
    color_val = torch.from_numpy(color_val)

    grey_start, grey_end = pc_args["grey_idx_to_prod_se_array"][grey_index]

    flow_val = flow[grey_start : grey_end, ]
    flow_val = flow_val.unsqueeze(-1) # [64, 1]
 
    result = flow_val * color_val
    column_sums = result.sum(dim=0)
    column_sums = column_sums / torch.sum(column_sums)
    
    max_column_index = torch.multinomial(column_sums, 1).item()
    candidate_color = torch.arange(0, 256)
    return candidate_color[max_column_index]

def is_left(index):
    # Calculate the column position
    column = index % 16
    
    # Determine if it's on the left or right side
    if column <= 7:
        return True
    else:
        return False

def color_patch_algo(pc, ns, args, pc_args, Grey_patch, grey_prob):
    start_time = time.perf_counter()
    grey_lls, backward_node_mars = forward(pc, ns, grey_prob, args)
    end_time = time.perf_counter()
    # print(f"Time taken by forward: {(end_time - start_time):.6f} seconds")

    # print(f"grey_lls: {grey_lls}")
    
    start_time = time.perf_counter()
    flow = backward(pc, backward_node_mars)
    end_time = time.perf_counter()
    # print(f"Time taken by backward: {(end_time - start_time):.6f} seconds")
    flow = flow.cpu()

    color_tensor = torch.zeros(args.data_shape)
    h = args.data_shape[1]
    w = args.data_shape[2]
    
    progress_bar = tqdm(total=args.data_shape[1]*args.data_shape[2], desc="Compute Color")
    for y in range(h):
        for z in range(w):
            grey_index = y * w + z
            grey = Grey_patch.flatten()[grey_index]

            if is_left(grey_index):
                sample_color = grey
            else:
                sample_color = sample_pixel(pc, ns, args, pc_args, flow, grey_index)
            color_tensor[0, y, z] = sample_color
            
            progress_bar.update(1)

    return color_tensor[0, :, :]

def update_grey_prob(grey_prob, color_tensor, grey_arr, grey_index, args, pc_args):
    if grey_index < 1:
        return grey_prob
    else:
        grey_idx_to_prod_se_arr = pc_args["grey_idx_to_prod_se_array"]
        param_arr = pc_args["param_array"]
        arr_valid_ycocg = pc_args["arr_valid_ycocg"]
        arr_num_valid_ycocg = pc_args["arr_num_valid_ycocg"]

        target_grey_start, target_grey_end = grey_idx_to_prod_se_arr[grey_index]
        grey_prob[target_grey_start : target_grey_end, :] = -np.inf
        for temp_grey_index, temp_grey in enumerate(grey_arr):
            grey_start, grey_end = grey_idx_to_prod_se_arr[temp_grey_index]
            if target_grey_start == grey_start and target_grey_end == grey_end:
                h = args.data_shape[1]
                w = args.data_shape[2]
                row = temp_grey_index // w
                col = temp_grey_index % w

                red_index = row * w + col
                green_index = h * w + row * w + col
                blue_index = 2 * h * w + row * w + col

                red_params = param_arr[red_index] # (64, 256)
                green_params = param_arr[green_index] # (64, 256)
                blue_params = param_arr[blue_index] # (64, 256)

                if color_tensor[0, row, col] == 0 and color_tensor[1, row, col] == 0 and color_tensor[2, row, col] == 0:
                    valid_ycocg = arr_valid_ycocg[temp_grey]
                    valid_ycocg = valid_ycocg[:, :int(arr_num_valid_ycocg[temp_grey])]

                    red_vals = red_params[:, np.array(valid_ycocg[0], dtype=int)] # (64, 13689)
                    green_vals = green_params[:, np.array(valid_ycocg[1], dtype=int)] # (64, 13689)
                    blue_vals = blue_params[:, np.array(valid_ycocg[2], dtype=int)] # (64, 13689)

                    grey_vals = red_vals*green_vals*blue_vals # (64, 13689)
                else:
                    red_vals = red_params[:, int(color_tensor[0, row, col])]
                    green_vals = green_params[:, int(color_tensor[1, row, col])]
                    blue_vals = blue_params[:, int(color_tensor[2, row, col])]

                    grey_vals = red_vals*green_vals*blue_vals
                    grey_vals = np.expand_dims(grey_vals, axis=-1)

                grey_vals_sum = np.sum(grey_vals, axis=1)
                grey_vals_sum = grey_vals_sum.reshape(-1, 1)
                grey_vals = np.log(grey_vals_sum)
                
                if np.all(grey_prob[grey_start : grey_end, :] == -np.inf):
                    grey_prob[grey_start : grey_end, :] = grey_vals
                else:
                    grey_prob[grey_start : grey_end, :] += grey_vals

    return grey_prob # (pc_num_elements, 1)

def color_patch_algo_auto(pc, ns, args, pc_args, Grey_patch, grey_prob, groun_truth_color):
    groun_truth_color = groun_truth_color.reshape(args.data_shape)
    groun_truth_color = groun_truth_color.cpu()

    color_tensor = torch.zeros(args.data_shape)
    h = args.data_shape[1]
    w = args.data_shape[2]

    arr_valid_ycocg = pc_args["arr_valid_ycocg"]
    arr_num_valid_ycocg = pc_args["arr_num_valid_ycocg"]
    
    progress_bar = tqdm(total=args.data_shape[0]*args.data_shape[1]*args.data_shape[2], desc="Compute Color")

    # sample_conditional_likelihood = 0
    for grey_index, grey in enumerate(Grey_patch.flatten()):
        # print(f"grey_index: {grey_index}, grey: {grey}")
        grey_prob = update_grey_prob(grey_prob, color_tensor, Grey_patch.flatten(), grey_index, args, pc_args)

        grey_lls, backward_node_mars = forward(pc, ns, grey_prob, args)
        # print(f"grey_lls: {grey_lls}")
        
        flow = backward(pc, backward_node_mars)
        flow = flow.cpu()

        y, z = grey_index_to_rgb_rc(grey_index, args)
        valid_ycocg = arr_valid_ycocg[grey]
        valid_ycocg = valid_ycocg[:, :int(arr_num_valid_ycocg[grey])]
        valid_ycocg = valid_ycocg.astype(np.int32) 
        valid_ycocg = torch.from_numpy(valid_ycocg) # [3, 25543]

        for c in range(args.data_shape[0]):
            candidate_color = torch.unique(valid_ycocg[c, :])
            groun_truth_color_pixel = groun_truth_color[c, y, z]

            color = ["red", "green", "blue"][c]
            if args.argmax:
                sample_color = sample_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color)
            else:
                sample_color = sample_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color)

            color_tensor[c, y, z] = sample_color
            valid_ycocg = valid_ycocg[:, torch.abs(valid_ycocg[c, :] - color_tensor[c, y, z]) < 10**(-2)]

            progress_bar.update(1)
    
    return color_tensor