# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:32:47 2023

@author: user
"""

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

class_names = ['-16', '-9', '-5', '-2', '0', '1', '4', '6', '10', '14', '23', '999']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (224, 224)

def load_data(test_size=0.2):
    DIRECTORY = r'C:\Users\user\Desktop\check_stock\stockdataV1\imagestock'
    CATEGORIES = ['-16', '-9', '-5', '-2', '0', '1', '4', '6', '10', '14', '23', '999']
    
    images = []
    labels = []
    
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
    
        print('loading {}'.format(category))
    
        for file in tqdm(os.listdir(path)):
            img_path = os.path.join(path, file)
            
            # Extract class label from category
            label = class_names_label[category]
    
            # Use PIL to open the image
            image = Image.open(img_path)
            image = image.convert("RGB")  # Ensure image has 3 channels
    
            # Resize the image
            image = image.resize(IMAGE_SIZE, Image.ANTIALIAS)
    
            # Convert PIL Image to NumPy array
            image = np.array(image, dtype='float32')
    
            images.append(image)
            labels.append(label)
    
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    
    # Split the data into train and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data(test_size=0.2)

# Define your model here
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(12, activation=tf.nn.softmax)  # Updated to 12 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=128, epochs=6, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model as .h5 file
model.save('keraV3.h5')

# Make predictions on the test data
test_predictions = model.predict(test_images)
test_predicted_labels = np.argmax(test_predictions, axis=1)

# Calculate the confusion matrix
confusion = confusion_matrix(test_labels, test_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
