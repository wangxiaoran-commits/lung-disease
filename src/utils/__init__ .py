from matplotlib import pyplot as plt
import numpy as np

def plot_line(title, xlabel, ylabel, train_label, test_label, data, test_data, times_test, save_path):
    plt.cla()
    plt.title(title)  # 图片标题
    plt.xlabel(xlabel)  # x轴变量名称
    plt.ylabel(ylabel)  # y轴变量名称
    plt.plot(np.arange(1, len(data) + 1, 1), data, label=train_label)  # 逐点画出trian_loss_results值并连线，连线图标是Loss
    plt.plot(np.arange(times_test, len(data) + 1, times_test), test_data, label=test_label)
    plt.legend()  # 画出曲线图标
    plt.savefig(save_path, dpi=350)
