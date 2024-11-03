import os
from PIL import Image

# Root directory where images are stored
root_dir = ""  # Update this to your image folder path

# Number of images (folders) to process
n_images = 30

# Image file names within each directory
image_types = ["Grey_Image.png", "Original_Image.png", "Pred_Image.png", "YCoCg_Image.png", "Pred_YCoCg_Image.png"]

# Initialize list to hold horizontally concatenated image strips
horizontal_strips = []

# Loop over the first 50 image directories (Image0, Image1, ..., Image49)
for i in range(n_images):
    # Construct the path for the current image directory (e.g., "Image0", "Image1", ...)
    image_dir = os.path.join(root_dir, f"sample2/Image{i}")
    
    # Ensure the directory exists
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist, skipping.")
        continue
    
    # Load each of the five image types from the current directory
    images = []
    for img_type in image_types:
        img_path = os.path.join(image_dir, img_type)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"Image {img_path} not found, skipping this image.")
            continue
    
    # Check if all five images were loaded
    if len(images) != 5:
        print(f"Some images are missing in {image_dir}, skipping this directory.")
        continue

    # Concatenate the images vertically (stack them on top of each other)
    width, height = images[0].size
    vertical_strip = Image.new('RGB', (width, height * 5))  # Create a blank canvas for concatenation
    
    for j, img in enumerate(images):
        vertical_strip.paste(img, (0, j * height))  # Paste each image at the correct vertical position

    # Add this vertical strip to the list for horizontal concatenation
    horizontal_strips.append(vertical_strip)

# Now concatenate all vertical strips horizontally
if horizontal_strips:
    total_width = width * len(horizontal_strips)  # Width for the final concatenated image
    total_height = height * 5  # Height is fixed as 5 times the individual image height
    
    final_image = Image.new('RGB', (total_width, total_height))  # Create a blank canvas for the final image
    
    # Paste each vertical strip into the final image
    for idx, strip in enumerate(horizontal_strips):
        final_image.paste(strip, (idx * width, 0))  # Position each strip side by side

    # Save the final concatenated image as a high-resolution image
    final_image.save('concatenated_image.png', quality=95)  # You can adjust the quality level as needed
    print("Final concatenated image saved as 'concatenated_image.png'")
else:
    print("No valid image strips were generated.")
