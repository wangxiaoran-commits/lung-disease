class MyConfig():
    def __init__(self):
        self.config_name = "p1_resunet"

        self.data_root_path = "/home/jovyan/work/datasets/66836d55111e5909c8b26658-662777f4b3377299d4034f74/COVID-19_Pre_Dataset"
        self.categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.data_batch_size = 16

        self.input_shape = (256, 256, 1)
        self.class_num = 4

        self.epoch = 10
        self.times_val = 1
        self.learn_rate = 1e-5

        self.base_save_path = r"/home/jovyan/work/results/"







