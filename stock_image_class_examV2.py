import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 载入模型
filepath1 = r'C:\Users\user\Desktop\check_stock\converted_keras'
model = keras.models.load_model(filepath1 + '/your_model_name.h5', compile=False)
# model = keras.models.load_model(filepath1 + '/keraV3.h5', compile=False)
# 定义标签映射字典
label_map = {
    0: -16,
    1: -9,
    2: -5,
    3: -2,
    4: 0,
    5: 1,
    6: 4,
    7: 6,
    8: 10,
    9: 14,
    10: 23,
    11: 999
}

# 定义函数进行图像分类
def classify_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label_idx = np.argmax(prediction)
    predicted_label = label_map[predicted_label_idx]
    return predicted_label

# 收集真实标签和预测标签
true_labels = []
predicted_labels = []

# 指定包含图像的文件夹路径
base_dir = r'C:\Users\user\Desktop\check_stock\image'
subfolders = os.listdir(base_dir)

# 对每个文件夹中的图像进行分类
for folder in subfolders:
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        image_files = os.listdir(folder_path)
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            true_label = int(folder)
            predicted_label = classify_image(image_path)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels, labels=list(label_map.values()))

    

# 计算准确率
accuracy = np.trace(cm) / np.sum(cm)
print("Accuracy:", accuracy)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
disp.plot(cmap='Blues', values_format='.0f')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 显示准确率
plt.text(0.5, -0.3, f'Accuracy: {accuracy:.2f}', size=12, ha="center", transform=plt.gca().transAxes)

plt.show()

