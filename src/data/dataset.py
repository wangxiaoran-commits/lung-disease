import tensorflow as tf
import numpy as np
import os
import random
import glob
from sklearn.model_selection import train_test_split
from myconfig import MyConfig


def load_and_preprocess_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [256, 256])
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask


def augment_image(image, mask):
    if random.random() > 0.5:
        image = tf.image.random_flip_left_right(image)
        mask = tf.image.random_flip_left_right(mask)
    if random.random() > 0.5:
        image = tf.image.random_flip_up_down(image)
        mask = tf.image.random_flip_up_down(mask)

    return image, mask


def load_image_labels(dataset_path, categories):
    image_paths = []
    mask_paths = []
    labels = []

    for i, category in enumerate(categories):
        category_images_path = os.path.join(dataset_path, category, 'images', '*.png')
        category_masks_path = os.path.join(dataset_path, category, 'masks', '*.png')

        images = sorted(glob.glob(category_images_path))
        masks = sorted(glob.glob(category_masks_path))

        combined = list(zip(images, masks, [i] * len(images)))
        random.shuffle(combined)
        images, masks, label_list = zip(*combined)

        image_paths.extend(images)
        mask_paths.extend(masks)
        labels.extend(label_list)

    return image_paths, mask_paths, labels


def create_dataset(image_paths, mask_paths, labels, batch_size, num_classes, do_augment=False):
    def generator():
        for img_path, mask_path, label in zip(image_paths, mask_paths, labels):
            yield img_path, mask_path, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.string, tf.string, tf.int32),
        output_shapes=((), (), ()),
    )

    def load_image(img_path, mask_path, label):
        image, mask = load_and_preprocess_image(img_path, mask_path)
        label = tf.one_hot(label, num_classes)
        if do_augment is True:
            image, mask = augment_image(image, mask)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset



if __name__ == '__main__':
    # Set paths
    myconfig = MyConfig()
    dataset_path = myconfig.data_root_path
    categories = myconfig.categories
    batch_size = myconfig.data_batch_size
    class_num = myconfig.class_num

    image_paths, mask_paths, labels = load_image_labels(dataset_path, categories)

    train_img_paths, val_test_img_paths, train_mask_paths, val_test_mask_paths, train_labels, val_test_labels = train_test_split(
        image_paths, mask_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    val_img_paths, test_img_paths, val_mask_paths, test_mask_paths, val_labels, test_labels = train_test_split(
        val_test_img_paths, val_test_mask_paths, val_test_labels, test_size=0.5, random_state=42,
        stratify=val_test_labels
    )

    train_dataset = create_dataset(train_img_paths, train_mask_paths, train_labels, batch_size, class_num,
                                   do_augment=True)
    val_dataset = create_dataset(val_img_paths, val_mask_paths, val_labels, batch_size, class_num, do_augment=False)
    test_dataset = create_dataset(test_img_paths, test_mask_paths, test_labels, batch_size, class_num, do_augment=False)

    print("Train, validation, and test datasets have been created.")

    # print(len([_ for _ in enumerate(train_dataset)]))
    # for i, (x, m, y) in enumerate(val_dataset):
    #     print("step 1")
    #     print(type(x))
    #     print(f"x:{x}")
    #     print(f"y:{y.shape}")
    #     print(f"m:{m}")
    #     print(f"max-m:{np.max(m)}")
    #     break
