import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from myconfig import MyConfig


def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png")):
                count += 1
    return count


def analyze_images(directory):
    sizes = []
    channels = []
    pixel_min_values = []
    pixel_max_values = []
    normalized_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png")):
                with Image.open(os.path.join(root, file)) as img:
                    if img.mode == 'RGB':
                        img = img.convert('L')
                    sizes.append(img.size)
                    channels.append(len(img.getbands()))
                    pixels = np.array(img)

                    norm_pixels = (pixels - pixels.mean()) / pixels.std()
                    normalized_images.append(norm_pixels)

                    pixel_min_values.append(pixels.min())
                    pixel_max_values.append(pixels.max())
    return sizes, channels, pixel_min_values, pixel_max_values, normalized_images


def show_image_samples(directory, category, num_samples=2):
    images = [Image.open(os.path.join(directory, f)) for f in os.listdir(directory) if f.lower().endswith(".png")]
    fig, axs = plt.subplots(ncols=num_samples, figsize=(8, 4))
    for idx, image in enumerate(np.random.choice(images, size=num_samples, replace=False)):
        axs[idx].imshow(image)
        axs[idx].axis('off')
        axs[idx].set_title(f"{category} Sample {idx + 1}")
    plt.show()

if __name__ == '__main__':
    myconfig = MyConfig()
    dataset_path = myconfig.data_root_path

    assert os.path.exists(dataset_path), "文件路径不存在"

    categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    category_counts = []
    for category in categories:
        category_path = os.path.join(dataset_path, category, 'images')
        mask_path = os.path.join(dataset_path, category, 'masks')

        num_images = count_images(category_path)
        num_masks = count_images(mask_path)
        category_counts.append(num_images)

        print(f"Category: {category}")
        print(f"Number of images: {num_images}")
        print(f"Number of masks: {num_masks}")

        sizes, channels, pixel_min_values, pixel_max_values, normalized_images = analyze_images(category_path)

        print(f"Image size: {sizes[0] if sizes else 'N/A'}")
        print(f"Channels: {channels[0] if channels else 'N/A'}")
        if pixel_min_values and pixel_max_values:
            print(f"Pixel value range (min): {np.min(pixel_min_values)}")
            print(f"Pixel value range (max): {np.max(pixel_max_values)}")
        else:
            print("Pixel value range: N/A")
        print("-" * 30)

        # 展示部分图片样本
        # show_image_samples(category_path, category)

    plt.figure(figsize=(10, 6))
    plt.bar(categories, category_counts, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Categories')
    plt.show()

    total_images = sum(category_counts)
    ratios = [count / total_images for count in category_counts]
    for category, ratio in zip(categories, ratios):
        print(f"Category: {category}, Ratio: {ratio:.2%}")