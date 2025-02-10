import os
from PIL import Image

def convert_tif_to_png(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            try:
                input_path = os.path.join(input_dir, filename)

                with Image.open(input_path) as img:
                    output_filename = os.path.splitext(filename)[0] + ".png"
                    output_path = os.path.join(output_dir, output_filename)

                    img.save(output_path, "PNG")

                print(f"Converted: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

input_directory = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG'  
output_directory = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_PNG'  
convert_tif_to_png(input_directory, output_directory)
