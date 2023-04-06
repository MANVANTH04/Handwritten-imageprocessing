#EPOCH 20 Only SMALL Letters, printing test data to see the samples
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

data_path = "/Users/nikhitha/Desktop/letters/train_data"

# initially trained data with folder name A containing all small 'a' inside it
alphabet_dict = {
    "A": 0,
    "B": 1,
    "Z": 2,
    "C":3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H":8,
    "I":9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M":13,
    "N":14,
    "O":15,
    "P":16,
    "Q":17,
    "R":18,
    "S":19,
    "T":20,
    "U":21,
    "V":22,
    "W":23,
    "X":24,
    "Y":25,
}

# Set the image size
img_size = 28

# Load the data and labels from the data folder
data = []
labels = []
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        label = alphabet_dict[folder]
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype("float32") / 255.0
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(str(e))
            

print(labels)
data = np.array(data)
labels = np.array(labels)


data_tensor = tf.convert_to_tensor(data)
labels_tensor = tf.convert_to_tensor(labels)
print(data_tensor)
print(labels_tensor)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(alphabet_dict), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("test_data")
print(test_data)
model.fit(train_data, train_labels, epochs=20, validation_data=(test_data, test_labels))

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test accuracy:", test_acc)
