import os
from PIL import Image
import pandas as pd
import tifffile

def split_image_into_25(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = Image.open(image_path)
    width, height = img.size
    part_width = width // 5
    part_height = height // 5
    image_paths = []
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    count = 0

    for i in range(5):
        for j in range(5):
            left = j * part_width
            upper = i * part_height
            right = (j + 1) * part_width
            lower = (i + 1) * part_height
            img_cropped = img.crop((left, upper, right, lower))
            part_filename = f"{base_filename}_part_{count}.tif"
            part_path = os.path.join(output_dir, part_filename)
            img_cropped.save(part_path)
            image_paths.append(part_path)
            count += 1

    df = pd.DataFrame({'Image_Parts': image_paths})
    csv_path = os.path.join(output_dir, 'image_parts.csv')
    df.to_csv(csv_path, index=False)

    print(f"Image split into 25 parts and saved in {output_dir} without metadata")
    return df

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            input_image_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            split_image_into_25(input_image_path, output_dir)

input_directory = "D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG"
output_directory = "D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_Output_tif"
process_directory(input_directory, output_directory)
