import os
import numpy as np
import rasterio
import csv

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

input_tif = 'D:/Project Caldermeade/NJR/Training/AG/2022-01-24/DCIM/104FPLAN_20/AG/DJI_0406.tif'

csv_file_path = os.path.join(output_dir, 'average_rgb_values_scaled.csv')

with rasterio.open(input_tif) as src:
    image_array = src.read()

    red = image_array[0]  
    green = image_array[1]  
    blue = image_array[2]  

    max_pixel_value = 65535  
    scale_factor = 255 / max_pixel_value

    avg_red = np.mean(red) * scale_factor
    avg_green = np.mean(green) * scale_factor
    avg_blue = np.mean(blue) * scale_factor

    avg_red = np.clip(avg_red, 0, 255)
    avg_green = np.clip(avg_green, 0, 255)
    avg_blue = np.clip(avg_blue, 0, 255)

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Average Red', 'Average Green', 'Average Blue'])

        writer.writerow([avg_red, avg_green, avg_blue])

print(f"Scaled average RGB values (0-255) saved to {csv_file_path}")
