import os
import datetime
from tqdm import tqdm
from models.resunet import ResUNet
from myconfig import MyConfig
import data.dataset as ds
import tensorflow as tf
import tensorflow.keras as K
from utils import *
import pandas as pd
from tensorflow import keras as K
from metric.f1_score import F1Score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class SaveBestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy and current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.model.save(self.save_path)
            print(f'\nEpoch {epoch+1}: val_accuracy improved to {current_val_accuracy:.4f}, saving model to {self.save_path}')



class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics_save_path):
        super().__init__()
        self.metrics_save_path = metrics_save_path
        self.train_metrics = []
        self.val_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_metrics.append({k: v for k, v in logs.items() if not k.startswith('val_')})
        self.val_metrics.append({k: v for k, v in logs.items() if k.startswith('val_')})

        train_df = pd.DataFrame(self.train_metrics)
        val_df = pd.DataFrame(self.val_metrics)

        # 保存到 Excel
        train_df.to_excel(os.path.join(self.metrics_save_path, 'train.xlsx'), index_label='epoch')
        val_df.to_excel(os.path.join(self.metrics_save_path, 'val.xlsx'), index_label='epoch')

        # 绘制折线图
        for metric in train_df.columns:
            plt.figure()
            plt.plot(train_df[metric], label=f'Train {metric}')
            if f'val_{metric}' in val_df.columns:
                plt.plot(val_df[f'val_{metric}'], label=f'Validation {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'Train and Validation {metric}')
            plt.legend()
            plt.savefig(os.path.join(self.metrics_save_path, f'{metric}.png'))
            plt.close()


def creat_model(load_path):
    config = MyConfig()
    if load_path is not None:
        print("model loading")
        model = K.models.load_model(load_path, compile=False)
        print("load finish")
    else:
        model = ResUNet(config.input_shape, config.class_num)
    model.compile(loss=K.losses.categorical_crossentropy,
                  optimizer=K.optimizers.Adam(learning_rate=config.learn_rate),
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           F1Score(name='f1_score')])
    return model


def trainer(train_dataset, val_dataset, load_path):
    config = MyConfig()
    model = creat_model(load_path)
    uid = datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S/{config.config_name}")

    epochs = config.epoch
    BATCH_SIZE = config.data_batch_size

    base_save_path = os.path.join(config.base_save_path, uid)
    model_save_path = os.path.join(base_save_path, 'model.h5')
    metrics_save_path = os.path.join(base_save_path, 'metrics')
    os.makedirs(metrics_save_path, exist_ok=True)

    metrics_callback = MetricsCallback(metrics_save_path)
    save_best_model_callback = SaveBestModelCallback(model_save_path)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        verbose=1,
        callbacks=[metrics_callback, save_best_model_callback])


def metrics_save(history, metrics_save_path):
    metrics = history.history
    train_metrics = {k: v for k, v in metrics.items() if not k.startswith('val_')}
    val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}

    # 创建数据框
    train_df = pd.DataFrame(train_metrics)
    val_df = pd.DataFrame(val_metrics)

    # 保存到 Excel
    train_df.to_excel(os.path.join(metrics_save_path, 'train.xlsx'), index_label='epoch')
    val_df.to_excel(os.path.join(metrics_save_path, 'val.xlsx'), index_label='epoch')

    # 绘制折线图
    for metric in train_metrics.keys():
        plt.figure()
        plt.plot(train_df[metric], label=f'Train {metric}')
        if f'val_{metric}' in val_metrics:
            plt.plot(val_df[f'val_{metric}'], label=f'Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Train and Validation {metric}')
        plt.legend()
        plt.savefig(os.path.join(metrics_save_path, f'{metric}.png'))
        plt.close()


def test_model(test_dataset, load_path, save_path):
    model = creat_model(load_path)

    # 预测整个测试集
    results = model.evaluate(test_dataset)
    metrics_names = model.metrics_names
    results_dict = dict(zip(metrics_names, results))

    # 保存整体测试结果
    with open(os.path.join(save_path, 'test_all_result.txt'), 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")

    # 绘制并保存混淆矩阵
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(model.predict(images), axis=1).numpy())

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for All Classes')
    plt.savefig(os.path.join(save_path, 'confusion_matrix_all.png'))
    plt.close()

    # 按类别评估
    config = MyConfig()
    class_names = config.categories
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, class_name in enumerate(class_names):
        idx = (y_true == i)
        class_y_true = y_true[idx]
        class_y_pred = y_pred[idx]

        # class_report = classification_report(class_y_true, class_y_pred, output_dict=True, zero_division=0)
        class_report = classification_report(class_y_true, class_y_pred, output_dict=True)
        with open(os.path.join(save_path, f'test_{class_name}_result.txt'), 'w') as f:
            for key, value in class_report.items():
                f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    device_num = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > device_num:
        tf.config.experimental.set_memory_growth(gpus[device_num], True)

    myconfig = MyConfig()
    dataset_path = myconfig.data_root_path
    categories = myconfig.categories
    batch_size = myconfig.data_batch_size
    class_num = myconfig.class_num

    image_paths, mask_paths, labels = ds.load_image_labels(dataset_path, categories)

    train_img_paths, val_test_img_paths, train_mask_paths, val_test_mask_paths, train_labels, val_test_labels = ds.train_test_split(
        image_paths, mask_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    val_img_paths, test_img_paths, val_mask_paths, test_mask_paths, val_labels, test_labels = ds.train_test_split(
        val_test_img_paths, val_test_mask_paths, val_test_labels, test_size=0.5, random_state=42,
        stratify=val_test_labels
    )

    train_dataset = ds.create_dataset(train_img_paths, train_mask_paths, train_labels, batch_size, class_num,
                                      do_augment=True)
    val_dataset = ds.create_dataset(val_img_paths, val_mask_paths, val_labels, batch_size, class_num, do_augment=False)
    test_dataset = ds.create_dataset(test_img_paths, test_mask_paths, test_labels, batch_size, class_num,
                                     do_augment=False)
    print("Train, validation, and test datasets have been created.")

    load_path = None
    # trainer(train_dataset, val_dataset, load_path)

    # 测试模型
    test_model_path = r"/home/jovyan/work/results/20240708_105439/p1_resunet/model.h5"
    test_save_path = r"/home/jovyan/work/results/test/20240708_105439"
    os.makedirs(test_save_path, exist_ok=True)
    test_model(test_dataset, test_model_path, test_save_path)
