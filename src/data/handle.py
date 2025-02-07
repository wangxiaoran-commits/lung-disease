import os
import numpy as np
from PIL import Image
from myconfig import MyConfig


def process_image(image_path):
    with Image.open(image_path) as img:
        if img.mode == 'RGB':
            img = img.convert('L')

        if img.size == (299, 299):
            left = int((299 - 256) / 2)
            top = 0
            right = int((299 + 256) / 2)
            bottom = 256
            img = img.crop((left, top, right, bottom))

        return img


def process_images_in_directory(source_dir, target_dir, mask_dir=None, fuse_dir=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if fuse_dir and not os.path.exists(fuse_dir):
        os.makedirs(fuse_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".png"):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                processed_image = process_image(source_path)
                processed_image.save(target_path)  

                if mask_dir:
                    mask_path = os.path.join(mask_dir, relative_path)
                    mask_target_path = os.path.join(target_dir.replace('images', 'masks'), relative_path)
                    if os.path.exists(mask_path):
                        processed_mask = process_image(mask_path)
                        os.makedirs(os.path.dirname(mask_target_path), exist_ok=True)
                        processed_mask.save(mask_target_path)  

                        if fuse_dir:
                            fuse_image = fuse_images(processed_image, processed_mask)
                            fuse_target_path = os.path.join(fuse_dir, relative_path)
                            os.makedirs(os.path.dirname(fuse_target_path), exist_ok=True)
                            fuse_image.save(fuse_target_path)  


def fuse_images(image, mask):
    image_np = np.array(image)
    mask_np = np.array(mask)

    fused_image_np = np.where(mask_np == 255, image_np, mask_np)

    return Image.fromarray(fused_image_np)


if __name__ == '__main__':

    myconfig = MyConfig()
    dataset_path = myconfig.data_root_path
    categories = myconfig.categories
    new_dataset_path = "E:/myproject/project/zjdx/dataset/COVID-19_Pre_Dataset"

    for category in categories:
        category_source_path = os.path.join(dataset_path, category, 'images')
        mask_source_path = os.path.join(dataset_path, category, 'masks')

        category_target_path = os.path.join(new_dataset_path, category, 'images')
        mask_target_path = os.path.join(new_dataset_path, category, 'masks')
        fuse_target_path = os.path.join(new_dataset_path, category, 'fuse')

        print(f"Processing images in category: {category}")
        process_images_in_directory(category_source_path, category_target_path, mask_source_path, fuse_target_path)
        print(f"Processing masks in category: {category}")
        process_images_in_directory(mask_source_path, mask_target_path)

    print("Image processing complete.")
