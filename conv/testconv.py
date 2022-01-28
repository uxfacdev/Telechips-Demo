import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def imageSetLoad():
    print("ImageSetLoad start")
    import os
    import cv2
    import numpy as np
    path = '../datasets-16/'
    trainPath = path + 'train/'
    testPath = path + 'test/'

    trainImageSet = []
    trainLabelSet = []
    testImageSet = []
    testLabelSet = []

    imageCategore = ['tin/', 'tinner/', 'tout/', 'touter/']
    imageLable = [0, 1, 2, 3]
    maxPixel = 255.0

    count = 0
    for categore in imageCategore:
        imageList = os.listdir(trainPath + categore)
        for file in imageList:
            image = cv2.imread(trainPath + categore + file)
            trainImageSet.append(image / maxPixel)
            trainLabelSet.append(imageLable[count])
        count += 1

    count = 0
    for categore in imageCategore:
        imageList = os.listdir(testPath + categore)
        for file in imageList:
            image = cv2.imread(testPath + categore + file)
            testImageSet.append(image / maxPixel)
            testLabelSet.append(imageLable[count])
        count += 1


    print("ImageSetLoad SetLoad end")
    trainImageSet = np.array(trainImageSet)
    trainLabelSet = np.array(trainLabelSet)
    testImageSet = np.array(testImageSet)
    testLabelSet = np.array(testLabelSet)

    print("ImageSetLoad End")

    return trainImageSet, trainLabelSet, testImageSet, testLabelSet


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='linear', padding='same', strides=(1, 1))
        # self.conv2 = Conv2D(32, (3, 3), activation='linear', padding='same')
        self.flatten = Flatten()
        # self.d1 = Dense(64, activation='linear')
        self.d2 = Dense(32, activation='linear')
        self.d3 = Dense(4, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.flatten(x)
        # x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = imageSetLoad()

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(5)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5)

    EPOCHS = 500

    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

    tf.keras.models.save_model(
        model, 'testweight', overwrite=True, include_optimizer=True, save_format='tf',
        signatures=None, options=None
    )

    import numpy as np

    writer = tf.summary.create_file_writer('./log/logs')
    with writer.as_default():
        for step in range(10):
            # other model code would go here
            tf.summary.scalar("my_metric", 0.5, step=step)
            tf.summary.image("my_images", x_test, step=step)
            writer.flush()
