import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2

class MODEL:
    def __init__(self):
        self.epochs, self.batrch_size, self.imageSize = 10, 8, (128,128)
        self.input_shape = self.imageSize + (3,)
        root_path = './label'
        
        print(self.input_shape)

        self.paths = np.random.permutation(glob.glob(root_path + "/*/*.jpg"))

        print("start loading")
        self.x_indep = np.array([cv2.resize(cv2.imread(path), self.imageSize, cv2.INTER_LINEAR) for path in self.paths])
        print("loaded")
        print(self.x_indep.shape)

        self.y_depen = np.array([path.split("\\")[-2] for path in self.paths])
        print(self.y_depen)
        self.k_classes = len(set(self.y_depen))
        print(self.k_classes)
        

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=self.k_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self):
        model = self.create_model()
        # model.summary()
        self.x_indep = self.x_indep.reshape(self.x_indep.shape[0], *self.input_shape)
        self.y_depen = pd.get_dummies(self.y_depen)
        print(self.x_indep.shape, self.y_depen.shape)
        model.fit(self.x_indep, self.y_depen, epochs=self.epochs, batch_size=self.batrch_size)
        model.save("model.h5")
        print("model saved")

if __name__ == "__main__":
    obj = MODEL()
    obj.train_model()
