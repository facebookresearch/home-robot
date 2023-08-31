from PIL import Image
import os
import numpy as np
import cv2

def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            height, width = img.shape[:2]

            if width < height:
                new_width = target_size
                new_height = int(target_size * (height / width))
            else:
                new_width = int(target_size * (width / height))
                new_height = target_size

            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized {filename} to {new_width}x{new_height}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_folder')
    # parser.add_argument('--reset',action='store_true')
    # parser.add_argument('--seed',default=0,type=int)
    args = parser.parse_args()
    # input_folder = "/Users/matthewchang/Desktop/airbnb_images/airbnb1"
    input_folder = args.input_folder
    output_folder = f"{input_folder}_resized"
    target_size = 512
    resize_images(input_folder, output_folder, target_size)
