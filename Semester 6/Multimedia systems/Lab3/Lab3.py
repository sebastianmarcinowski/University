import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from io import BytesIO
from docx import Document
from docx.shared import Inches
import scipy.fftpack

# Zadanie 1
def nearest_neighbor(image, scale_multiplier):
    org_height, org_width = image.shape[:2]
    new_width = int(org_width * scale_multiplier)
    new_height = int(org_height * scale_multiplier)
    scale_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = int(round(i * org_height / new_height))
            y = int(round(j * org_width / new_width))
            x = min(x, org_height - 1)
            y = min(y, org_width - 1)
            scale_image[i, j] = image[x, y]
    return scale_image

def bilinear_interpolation(image, scale_multiplier):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_multiplier)
    new_height = int(original_height * scale_multiplier)
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = i * (original_height - 1) / (new_height - 1)
            y = j * (original_width - 1) / (new_width - 1)
            x1 = int(np.floor(x))
            x2 = min(x1 + 1, original_height - 1)
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, original_width - 1)
            a = x - x1
            b = y - y1
            resized_image[i, j] = (1 - a) * (1 - b) * image[x1, y1] + a * (1 - b) * image[x2, y1] + (1 - a) * b * image[x1, y2] + a * b * image[x2, y2]
    return resized_image

def average_size_reduction(image, scale_multiplier):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_multiplier)
    new_height = int(original_height * scale_multiplier)
    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_multiplier
            y = j / scale_multiplier
            x_start = max(0, int(x - 1))
            x_end = min(original_height, int(x + 2))
            y_start = max(0, int(y - 1))
            y_end = min(original_width, int(y + 2))
            neighborhood = image[x_start:x_end, y_start:y_end]
            resized_image[i, j] = np.mean(neighborhood, axis=(0, 1))
    return resized_image

def weighted_average_size_reduction(image, scale_multiplier):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_multiplier)
    new_height = int(original_height * scale_multiplier)

    weights = np.array([[0.1, 0.2, 0.1],
                        [0.2, 0.2, 0.2],
                        [0.1, 0.2, 0.1]])
    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_multiplier
            y = j / scale_multiplier
            x_start = max(0, int(x - 1))
            x_end = min(original_height, int(x + 2))
            y_start = max(0, int(y - 1))
            y_end = min(original_width, int(y + 2))
            neighborhood = image[x_start:x_end, y_start:y_end]
            if neighborhood.shape[0] == 3 and neighborhood.shape[1] == 3:
                if len(image.shape) == 3:
                    resized_image[i, j] = np.sum(neighborhood * weights[:, :, np.newaxis], axis=(0, 1)) / np.sum(weights)
                else:
                    resized_image[i, j] = np.sum(neighborhood * weights) / np.sum(weights)
            else:
                resized_image[i, j] = np.mean(neighborhood, axis=(0, 1))
    return resized_image

def median_size_reduction(image, scale_multiplier):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_multiplier)
    new_height = int(original_height * scale_multiplier)
    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype=image.dtype)
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_multiplier
            y = j / scale_multiplier
            x_start = max(0, int(x - 1))
            x_end = min(original_height, int(x + 2))
            y_start = max(0, int(y - 1))
            y_end = min(original_width, int(y + 2))
            neighborhood = image[x_start:x_end, y_start:y_end]
            resized_image[i, j] = np.median(neighborhood, axis=(0, 1))
    return resized_image

def detect_edges(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to detect_edges")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def display_results(original, results, title):
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(title, fontsize=16)

    algorithms = ['Nearest Neighbor', 'Bilinear', 'Average', 'Weighted Average', 'Median']
    for i, (alg_name, (scaled, edges)) in enumerate(results.items()):
        axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original - {alg_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Scaled - {alg_name}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(edges, cmap="gray")
        axes[i, 2].set_title(f"Edges - {alg_name}")
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def display_small_results(original, results, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=16)

    algorithms = ['Nearest Neighbor', 'Bilinear']
    for i, (alg_name, (scaled, edges)) in enumerate(results.items()):
        axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original - {alg_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Scaled - {alg_name}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(edges, cmap="gray")
        axes[i, 2].set_title(f"Edges - {alg_name}")
        axes[i, 2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Eksperymenty
image_big1 = cv2.imread('IMG_BIG/IMG_BIG/BIG_0002.jpg')
image_big2 = cv2.imread('IMG_BIG/IMG_BIG/BIG_0003.jpg')
image_small1 = cv2.imread('IMG_SMALL/IMG_SMALL/SMALL_0002.png')
image_small2 = cv2.imread('IMG_SMALL/IMG_SMALL/SMALL_0003.png')
image_small3 = cv2.imread('IMG_SMALL/IMG_SMALL/SMALL_0004.jpg')
image_small4 = cv2.imread('IMG_SMALL/IMG_SMALL/SMALL_0005.jpg')

fragments = {
    "Fragment 1 BIG_0002": image_big1[0:500, 0:500],
    "Fragment 2 BIG_0002": image_big1[300:800, 1800:2300],
    "Fragment 3 BIG_0002": image_big1[2000:2500, 700:1200],
    "Fragment 4 BIG_0002": image_big1[1100:1600, 2600:3100],
    "Fragment 1 BIG_0003": image_big2[1750:2250, 0:500],
    "Fragment 2 BIG_0003": image_big2[1750:2250, 500:1000],
    "Fragment 3 BIG_0003": image_big2[1750:2250, 3000:3500],
    "Fragment 4 BIG_0003": image_big2[1750:2250, 1500:2000]
}
small_images = {
    "SMALL_0002.png": image_small1,
    "SMALL_0003.png": image_small2,
    "SMALL_0004.jpg": image_small3,
    "SMALL_0005.jpg": image_small4
}

big_images_scales = [0.1, 0.2, 0.3, 0.7]
small_images_scales = [1.5, 2, 2.5 , 5]

for name, fragment in fragments.items():
    for scale in big_images_scales:
        scaled_nearest = nearest_neighbor(fragment, scale)
        edges_nearest = detect_edges(scaled_nearest)

        scaled_bilinear = bilinear_interpolation(fragment, scale)
        edges_bilinear = detect_edges(scaled_bilinear)

        scaled_average = average_size_reduction(fragment, scale)
        edges_average = detect_edges(scaled_average)

        scaled_weighted_average = weighted_average_size_reduction(fragment, scale)
        edges_weighted_average = detect_edges(scaled_weighted_average)

        scaled_median = median_size_reduction(fragment, scale)
        edges_median = detect_edges(scaled_median)

        results = {
            'Nearest Neighbor': (scaled_nearest, edges_nearest),
            'Bilinear': (scaled_bilinear, edges_bilinear),
            'Average': (scaled_average, edges_average),
            'Weighted Average': (scaled_weighted_average, edges_weighted_average),
            'Median': (scaled_median, edges_median)
        }

        display_results(fragment, results, f"{name} - {scale}x")

for name, image in small_images.items():
    for scale in small_images_scales:
        scaled_nearest = nearest_neighbor(image, scale)
        edges_nearest = detect_edges(scaled_nearest)

        scaled_bilinear = bilinear_interpolation(image, scale)
        edges_bilinear = detect_edges(scaled_bilinear)

        results = {
            'Nearest Neighbor': (scaled_nearest, edges_nearest),
            'Bilinear': (scaled_bilinear, edges_bilinear)
        }

        display_small_results(image, results, f"{name} - {scale}x")