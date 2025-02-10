import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
import pandas as pd

# Calculate VARI (Vegetation Adjusted Reflectance Index)
def calculate_vari(red, green, blue):
    return (green - red) / (green + red - blue + 1e-10)  # Small epsilon added to avoid division by zero

# Apply edge detection to the VARI thresholded image
def apply_edge_detection(image, vari_thresholded):
    edges = filters.sobel(image)
    return vari_thresholded * (1 - edges)

# Create a custom colormap for visualization
def create_custom_colormap():
    colors = ['blue', 'green', 'yellow', 'red']
    return plt.cm.colors.LinearSegmentedColormap.from_list("custom_vari", colors)

# Apply custom thresholds to VARI values for segmentation
def apply_custom_thresholds(vari_normalized):
    thresholds = np.zeros_like(vari_normalized)
    thresholds[vari_normalized < 0.25] = 0  # Low vegetation
    thresholds[(vari_normalized >= 0.25) & (vari_normalized < 0.6)] = 0.5  # Medium vegetation
    thresholds[vari_normalized >= 0.6] = 1  # High vegetation
    return thresholds

# Save RGB channels as separate images
def save_rgb_images(red, green, blue, base_filename, output_rgb_directory):
    red_img = Image.fromarray((red).astype(np.uint8))
    green_img = Image.fromarray((green).astype(np.uint8))
    blue_img = Image.fromarray((blue).astype(np.uint8))

    red_img.save(os.path.join(output_rgb_directory, base_filename + '_red.png'))
    green_img.save(os.path.join(output_rgb_directory, base_filename + '_green.png'))
    blue_img.save(os.path.join(output_rgb_directory, base_filename + '_blue.png'))

# Process a single image and calculate VARI
def process_image(input_path, output_heatmap_directory, output_rgb_directory):
    try:
        # Load the image and split into RGB channels
        img = Image.open(input_path).convert("RGB")
        img = np.array(img).astype(float)
        red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Calculate the VARI
        vari = calculate_vari(red, green, blue)
        vari = np.clip(vari, -1, 1)  # Clip VARI between -1 and 1

        # Normalize VARI for thresholding
        vari_normalized = (vari - np.min(vari)) / (np.max(vari) - np.min(vari) + 1e-10)

        # Apply thresholds for visualization purposes
        thresholds = np.zeros_like(vari_normalized)
        thresholds[vari_normalized < 0.4] = 0  # Low vegetation
        thresholds[(vari_normalized >= 0.4) & (vari_normalized < 0.7)] = 0.5  # Medium vegetation
        thresholds[vari_normalized >= 0.7] = 1  # High vegetation

        # Apply edge detection
        edges = filters.sobel(vari_normalized)
        thresholds_with_edges = thresholds * (1 - edges)

        # Save the heatmap
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        heatmap_output_path = os.path.join(output_heatmap_directory, base_filename + '_heatmap.png')

        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = create_custom_colormap()
        im = ax.imshow(thresholds_with_edges, cmap=cmap, vmin=0, vmax=1)

        plt.axis('off')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('VARI (Vegetation Coverage)')
        plt.savefig(heatmap_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save RGB channels
        save_rgb_images(red, green, blue, base_filename, output_rgb_directory)

        # Calculate average RGB values and VARI
        avg_red = np.mean(red)
        avg_green = np.mean(green)
        avg_blue = np.mean(blue)
        avg_vari = np.mean(vari_normalized)

        # Calculate the proportions of pixels in each VARI range
        total_pixels = vari_normalized.size
        low_veg_proportion = np.sum(vari_normalized < 0.4) / total_pixels
        medium_veg_proportion = np.sum((vari_normalized >= 0.4) & (vari_normalized < 0.7)) / total_pixels
        high_veg_proportion = np.sum(vari_normalized >= 0.7) / total_pixels

        # Return heatmap path, averages, and vegetation proportions
        return (heatmap_output_path, avg_red, avg_green, avg_blue, avg_vari,
                low_veg_proportion, medium_veg_proportion, high_veg_proportion)

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None, None

# Process all images in the input directory
def process_images_in_directory(input_directory, output_heatmap_directory, output_rgb_directory):
    if not os.path.exists(output_heatmap_directory):
        os.makedirs(output_heatmap_directory)

    if not os.path.exists(output_rgb_directory):
        os.makedirs(output_rgb_directory)

    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    if not png_files:
        print("No .png files found in the directory.")
        return

    data = []  # Store results for each image

    for png_file in png_files:
        input_path = os.path.join(input_directory, png_file)

        print(f"Processing: {png_file}")
        output_path, avg_red, avg_green, avg_blue, avg_vari, low_veg_proportion, medium_veg_proportion, high_veg_proportion = process_image(
            input_path, output_heatmap_directory, output_rgb_directory)

        if avg_red is not None and avg_green is not None and avg_blue is not None:
            data.append([png_file, avg_red, avg_green, avg_blue, avg_vari, low_veg_proportion, medium_veg_proportion, high_veg_proportion])

    # Create a DataFrame to store the results
    df = pd.DataFrame(data, columns=['Image Name', 'Average Red', 'Average Green', 'Average Blue', 'Average VARI', 
                                     'Low Vegetation Proportion', 'Medium Vegetation Proportion', 'High Vegetation Proportion'])

    # Save the results to an Excel file
    excel_output_path = os.path.join(output_heatmap_directory, 'BG_rgb_vari_sliced_values_with_proportions.xlsx')
    df.to_excel(excel_output_path, index=False)

    print(f"RGB and VARI values with proportions saved to {excel_output_path}")

if __name__ == "__main__":
    input_directory = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_PNG' 
    output_heatmap_directory = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_output_heatmaps_updated'  
    output_rgb_directory = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_output_rgb_channels_updated'  

    process_images_in_directory(input_directory, output_heatmap_directory, output_rgb_directory)
