import numpy as np
import tensorflow.keras as keras
import cv2



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, pth_hr, pth_lr, batch_size=8, dim=(128, 128), n_channels=3,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.pth_hr = pth_hr,
        self.pth_lr = pth_lr,
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            id_split = ID.split("_")
            hr_img = cv2.imread(self.pth_hr[0] + id_split[0])
            lr_img = cv2.imread(self.pth_lr[0] + id_split[0].split(".")[0] + "x4m.png")
            res_lr_img = cv2.resize(lr_img, (lr_img.shape[1] * 4, lr_img.shape[0] * 4))
            hr_patch = (hr_img[int(id_split[2]) * 64:(int(id_split[2]) * 64 + 128),
                        int(id_split[1]) * 128:(int(id_split[1]) * 128 + 128), :]) / 255.0
            lr_patch = (res_lr_img[int(id_split[2]) * 64:(int(id_split[2]) * 64 + 128),
                        int(id_split[1]) * 128:(int(id_split[1]) * 128 + 128), :]) / 255.0

            # Store sample
            X[i,] = lr_patch

            y[i,] = hr_patch

        return X, y
