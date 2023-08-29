from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = Image.open(input_path)
            width, height = img.size

            if width < height:
                new_width = target_size
                new_height = int(target_size * (height / width))
            else:
                new_width = int(target_size * (width / height))
                new_height = target_size

            resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
            resized_img.save(output_path)
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
