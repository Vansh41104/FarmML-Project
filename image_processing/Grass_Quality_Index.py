import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import os

input_dir = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14'
output_dir = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_output_index'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def classify_vari(vari_value):
    if vari_value >= 0.8:
        return 3  
    elif vari_value >= 0.6:
        return 2  
    elif vari_value >= 0.4:
        return 1  
    else:
        return 0  

def calculate_texture_features(image, window_size=5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_map = np.zeros_like(gray_image, dtype=np.float32)
    
    for i in range(window_size, gray_image.shape[0] - window_size):
        for j in range(window_size, gray_image.shape[1] - window_size):
            window = gray_image[i - window_size:i + window_size, j - window_size:j + window_size]
            texture_map[i, j] = window.var()  
    return texture_map

for file in os.listdir(os.path.join(input_dir, 'BG_output_heatmaps_updated')):
    if file.endswith('_heatmap.png'):  
        vari_image_path = os.path.join(input_dir, 'BG_output_heatmaps_updated', file)
        rgb_image_path = os.path.join(input_dir, 'BG_PNG', file.replace('_heatmap.png', '.png'))  

        vari_image = cv2.imread(vari_image_path)
        rgb_image = cv2.imread(rgb_image_path)

        if vari_image is None or rgb_image is None:
            print(f"Error: Could not load images for {file}. Skipping...")
            continue

        vari_gray = cv2.cvtColor(vari_image, cv2.COLOR_BGR2GRAY)

        vari_normalized = cv2.normalize(vari_gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

        grass_quality_map = np.vectorize(classify_vari)(vari_normalized)

        rgb_image_rescaled = exposure.rescale_intensity(rgb_image, in_range=(0, 255))

        texture_map = calculate_texture_features(rgb_image_rescaled)

        texture_map_normalized = cv2.normalize(texture_map, None, 0.0, 1.0, cv2.NORM_MINMAX)

        texture_map_resized = cv2.resize(texture_map_normalized, (vari_normalized.shape[1], vari_normalized.shape[0]))

        alpha = 0.7  
        beta = 0.3   

        grass_quality_index = alpha * vari_normalized + beta * texture_map_resized

        grass_quality_index_normalized = cv2.normalize(grass_quality_index.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title('VARI (Vegetation Index)')
        plt.imshow(vari_normalized, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title('Texture Map')
        plt.imshow(texture_map_resized, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title('Grass Quality Index')
        plt.imshow(grass_quality_index_normalized, cmap='jet')
        plt.colorbar()

        plot_output_path = os.path.join(output_dir, f'{file.replace("_heatmap.png", "")}_quality_plot.png')
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')

        index_output_path = os.path.join(output_dir, f'{file.replace("_heatmap.png", "")}_quality_index.png')
        cv2.imwrite(index_output_path, (grass_quality_index_normalized * 255).astype(np.uint8))

        print(f"Processed {file}:")
        print(f"Grass quality index image saved at: {index_output_path}")
        print(f"Grass quality plot saved at: {plot_output_path}")
