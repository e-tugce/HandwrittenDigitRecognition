import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

class CNN_Model:
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, train_labels, epochs, batch_size):
        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

cnn_model = CNN_Model()
cnn_model.train(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

random_index1 = np.random.randint(0, len(test_images))

sample_image1 = test_images[random_index1].reshape((1, 28, 28, 1))
prediction1 = cnn_model.model.predict(sample_image1)
predicted_label1 = tf.argmax(prediction1, axis=1).numpy()[0]
print(f"Predicted Label: {predicted_label1}")

plt.imshow(test_images[random_index1].reshape((28, 28)), cmap='gray')
plt.title(f"True Label: {tf.argmax(test_labels[random_index1]).numpy()}, Predicted Label: {predicted_label1}")
plt.show()