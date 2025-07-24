import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns

def MSE(source_image, modified_image):
    if source_image.shape != modified_image.shape:
        if len(modified_image.shape) == 2:  # Grayscale image
            modified_image = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2BGR)
    m, n = source_image.shape[:2]
    mse = (1 / (m * n)) * np.sum((source_image - modified_image) ** 2)
    return mse

def NMSE(source_image, modified_image):
    m = source_image.shape[0]
    n = source_image.shape[1]
    mse = MSE(source_image, modified_image)
    mse2 = MSE(source_image, np.zeros((m, n), dtype=np.uint8))
    nmse = mse / mse2
    return nmse

def PSNR(source_image, modified_image):
    mse = MSE(source_image, modified_image)
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value**2) / mse)
    return psnr

def IF (source_image, modified_image):
    if_numerator = np.sum((source_image-modified_image)**2)
    if_denumerator = np.sum(source_image*modified_image)
    return if_numerator/if_denumerator

def SSIM(source_image, modified_image):
    img1_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def JPEG_compression(file_name, image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    fig, axs = plt.subplots(1, 2, sharey=True)
    plt.suptitle(f'{file_name}-JPEG compression-Quality {quality}')
    axs[0].imshow(image)
    axs[1].imshow(decimg)
    # cv2.imwrite(f'{file_name}_JPEGCompress.png', decimg) #zapisanie obrazu po kompresji
    return decimg

def image_blur(file_name, image, kernel_size, sigX, sigY):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigX, sigY)
    fig, axs = plt.subplots(1, 2, sharey=True)
    plt.suptitle(f'{file_name}-Gaussian blur')
    axs[0].imshow(image)
    axs[1].imshow(blurred_image)
    # cv2.imwrite(f'{file_name}_GaussBlur.png', blurred_image) #zapisanie obrazu po rozmyciu
    return blurred_image

def image_noise(file_name, image, alfa, sigma):
    gauss = np.random.normal(0, sigma, image.shape)
    noise = (image + alfa * gauss).clip(0,255).astype(np.uint8)
    fig, axs = plt.subplots(1, 2, sharey=True)
    plt.suptitle(f'{file_name}-Gaussian noise')
    axs[0].imshow(image)
    axs[1].imshow(noise)
    # cv2.imwrite(f'{file_name}_Noise.png', noise) #zapisanie obrazu po dodaniu szumu
    return noise


def calculate_metrics(original, modified):
    return {
        "MSE": MSE(original, modified),
        "NMSE": NMSE(original, modified),
        "PSNR": PSNR(original, modified),
        "IF": IF(original, modified),
        "SSIM": SSIM(original, modified)
    }

original_image1 = cv2.imread('obraz1.jpg')
original_image2 = cv2.imread('obraz2.jpg')
original_image3 = cv2.imread('obraz3.jpg')
original_image4 = cv2.imread('obraz4.jpg')
results_df = pd.DataFrame(columns=["Plik", "Parametr", "MSE", "NMSE", "PSNR", "IF", "SSIM"])

# Add a separator row at the end of each iteration
for i in range(10):
    if i < 5:
        continue

    # Separator row
    # results_df.loc[len(results_df)] = [f"Iteration {i + 1}", "", "", "", "", "", ""]

    # JPEG Compression for image1
    quality = 10 + (5 * i)
    image1_mod = JPEG_compression(f'image1_iter{i}', original_image1, quality)
    metrics1 = calculate_metrics(original_image1, image1_mod)
    results_df.loc[len(results_df)] = ["image1", f"quality={quality}", *metrics1.values()]

    # Gaussian Blur for image2
    ksize = 5 + 2 * i  # e.g., 5, 7, 9...
    image2_mod = image_blur(f'image2_iter{i}', original_image2, ksize, 3, 3)
    metrics2 = calculate_metrics(original_image2, image2_mod)
    results_df.loc[len(results_df)] = ["image2", f"kernel={ksize}", *metrics2.values()]

    # Gaussian Noise for image3
    alfa = 0.1 * (i + 1)  # e.g., 0.1, 0.2, ...
    image3_mod = image_noise(f'image3_iter{i}', original_image3, alfa, 18)
    metrics3 = calculate_metrics(original_image3, image3_mod)
    results_df.loc[len(results_df)] = ["image3", f"alfa={round(alfa,2)}", *metrics3.values()]

metrics = ["MSE", "NMSE", "PSNR", "IF", "SSIM"]
# Plot each metric individually
plt.figure(figsize=(15, 10))

# Group by the "Plik" column
grouped = results_df.groupby("Plik")

for image_name, group in grouped:
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        plt.plot(group[metric].astype(float), marker='o', label=metric)
        plt.xlabel("Index")
        plt.ylabel(metric)
        plt.title(f"{metric} Over Iterations for {image_name}")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

# print(results_df)
# results_df.to_excel('results.xlsx', index=False)