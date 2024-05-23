import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import corner_harris
from skimage.measure import shannon_entropy

def calculate_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = np.sqrt((std_rg ** 2) + (std_yb ** 2)) + 0.3 * ((mean_rg + mean_yb) / 2.0)
    return colorfulness


def calculate_edge_density(gray_image):
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])


def calculate_sharpness(gray_image):
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(gray_image, cv2.CV_64F)))


folder_path = r"C:\Users\13729\Desktop\0430dataset\0430dataset"
data = []
accepted_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    file_extension = os.path.splitext(file)[1].lower()

    if os.path.isfile(file_path) and file_extension in accepted_extensions:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_r = np.mean(image[:, :, 2])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 0])

        dominant_color_index = np.bincount(image.reshape(-1, 3).argmax(1)).argmax()
        (dom_b, dom_g, dom_r) = image.reshape(-1, 3)[image.reshape(-1, 3).argmax(1) == dominant_color_index][0]

        colorfulness = calculate_colorfulness(image)
        entropy = shannon_entropy(image)

        skimage_image = io.imread(file_path)
        if skimage_image.shape[2] == 4:
            skimage_image = skimage_image[:, :, :3]

        harris = np.sum(corner_harris(rgb2gray(skimage_image)) > 0.01)
        edge_density = calculate_edge_density(gray_image)
        sharpness = calculate_sharpness(gray_image)

        is_banksy = 1 if "banksy" in file.lower() else 0

        data.append(
            [file, avg_r, avg_g, avg_b, dom_r, dom_g, dom_b, colorfulness, entropy, harris, edge_density, sharpness, is_banksy])

columns = ['image', 'avg_r', 'avg_g', 'avg_b', 'dom_r', 'dom_g', 'dom_b', 'colorfulness', 'entropy', 'harris', 'edge_density', 'sharpness', 'is_banksy']
df = pd.DataFrame(data, columns=columns)
df.to_csv('artwork_features.csv', index=False, encoding='utf-8')

print("CSV文件已生成！")
