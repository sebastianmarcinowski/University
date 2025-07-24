import numpy as np
import matplotlib.pyplot as plt
import cv2

def colorFit(color_value, color_palette):
    return color_palette[np.argmin(np.linalg.norm(color_palette - np.array(color_value), axis=1))]

def kwant_colorFit(img, pallet):
    out_img = img.copy().astype(np.float32) / 255.0
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            out_img[w, k] = colorFit(out_img[w, k], pallet)
    return out_img

M2 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
])
M4 = np.array([
    [0,32,8,40,2,34,10,42],
    [48,16,56,24,50,18,58,26],
    [12,44,4,36,14,46,6,38],
    [60,28,52,20,62,30,54,22],
    [3,35,11,43,1,33,9,41],
    [51,19,59,27,49,17,57,25],
    [15,47,7,39,13,45,5,37],
    [63,31,55,23,61,29,53,21]
])

def bayer_matrix(n):
    if n==1:
        return np.array([[0]])
    else:
        first = ((2*n) ** 2) * bayer_matrix(int(n / 2))
        second = ((2*n) ** 2) * bayer_matrix(int(n / 2)) + 2
        third = ((2*n) ** 2) * bayer_matrix(int(n / 2)) + 3
        fourth = ((2*n) ** 2) * bayer_matrix(int(n / 2)) + 1
        first_col = np.concatenate((first, third), axis=0)
        second_col = np.concatenate((second, fourth), axis=0)
        return np.concatenate((first_col, second_col), axis=1)

def random_dithering(img, palette):
    r = np.random.rand(img.shape[0], img.shape[1])
    out_img = img.copy().astype(np.float32) / 255.0
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            if out_img[w, k, 0] >= r[w, k]:  # Use the first channel for comparison
                out_img[w, k] = 1
            else:
                out_img[w, k] = 0
    return (out_img * 255).astype(np.uint8)

def organized_dithering(img, palette, m_matrix, n):
    Mpre = (m_matrix + 1) / ((2*n)**2) - 0.5
    # print(m_matrix)
    # print(Mpre)
    out_img = img.copy().astype(np.float32) / 255.0
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            out_img[w, k] = colorFit(out_img[w, k] + Mpre[w % (2*n), k % (2*n)], palette)
    return (out_img * 255).astype(np.uint8)

def Floyd_Steinberg_dithering(img, palette):
    out_img = img.copy().astype(np.float32) / 255.0
    height, width, _ = out_img.shape
    for w in range(out_img.shape[0]):
        for k in range(out_img.shape[1]):
            old_pixel = out_img[w, k].copy()
            new_pixel = colorFit(old_pixel, palette)
            out_img[w, k] = new_pixel
            quant_error = old_pixel - new_pixel
            if k + 1 < width:
                out_img[w, k + 1] += quant_error * 7 / 16
            if w + 1 < height:
                if k > 0:
                    out_img[w + 1, k - 1] += quant_error * 3 / 16
                out_img[w + 1, k] += quant_error * 5 / 16
                if k + 1 < width:
                    out_img[w + 1, k + 1] += quant_error * 1 / 16
    return (out_img * 255).astype(np.uint8)

palette1 = np.linspace(0,1, 2).reshape(2,1)
palette2 = np.linspace(0,1, 4).reshape(4,1)
palette4 = np.linspace(0,1,16).reshape(16,1)
palette8 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [1.0, 0.0, 0.0,],
        [1.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0,],
        [1.0, 1.0, 1.0,],
])
palette16 =  np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [0.0, 0.0, 1.0,],
        [1.0, 0.0, 1.0,],
        [0.0, 0.5, 0.0,],
        [0.5, 0.5, 0.5,],
        [0.0, 1.0, 0.0,],
        [0.5, 0.0, 0.0,],
        [0.0, 0.0, 0.5,],
        [0.5, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [1.0, 0.0, 0.0,],
        [0.75, 0.75, 0.75,],
        [0.0, 0.5, 0.5,],
        [1.0, 1.0, 1.0,],
        [1.0, 1.0, 0.0,]
])

# Wczytanie obrazów
img_gs1 = cv2.imread('IMG_GS/GS_0001.tif')
img_gs2 = cv2.imread('IMG_GS/GS_0002.png')
img_gs3 = cv2.imread('IMG_GS/GS_0003.png')
img_small1 = cv2.imread('IMG_SMALL/SMALL_0006.jpg')
img_small2 = cv2.imread('IMG_SMALL/SMALL_0007.jpg')
img_small3 = cv2.imread('IMG_SMALL/SMALL_0009.jpg')
img_small4 = cv2.imread('IMG_SMALL/SMALL_0010.jpg')

gs_images = {
    "GS_0001.tif": img_gs1,
    "GS_0002.png": img_gs2,
    "GS_0003.png": img_gs3
}
small_images = {
    "SMALL_0006.jpg": img_small1,
    "SMALL_0007.jpg": img_small2,
    "SMALL_0009.jpg": img_small3,
    "SMALL_0010.jpg": img_small4
}


for name, img in gs_images.items():
    plt.suptitle(name + " - 1 bit")
    plt.subplot(2,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Oryginał')
    plt.axis('off')

    img_kwant = kwant_colorFit(img, palette1)
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(img_kwant, cv2.COLOR_BGR2RGB))
    plt.title('Kwantyzacja')
    plt.axis('off')

    img_dith = random_dithering(img, palette1)
    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(img_dith, cv2.COLOR_BGR2RGB))
    plt.title('Dithering losowy')
    plt.axis('off')

    img_dith = organized_dithering(img, palette1, M2, 2)
    plt.subplot(2,3,4)
    plt.imshow(cv2.cvtColor(img_dith, cv2.COLOR_BGR2RGB))
    plt.title('Dythering zorganizowany')
    plt.axis('off')

    img_dith= Floyd_Steinberg_dithering(img, palette1)
    plt.subplot(2,3,6)
    plt.imshow(cv2.cvtColor(img_dith, cv2.COLOR_BGR2RGB))
    plt.title('Dythering Floyda-Steinberga')
    plt.axis('off')

    plt.show()

palettes = {
    "Palette 2": np.linspace(0, 1, 4).reshape(4, 1),
    "Palette 4": np.linspace(0,1,16).reshape(16,1)}
for name, img in gs_images.items():
    for palette_name, palette in palettes.items():
        plt.suptitle(f"{name} - {palette_name}")
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Oryginał')
        plt.axis('off')

        img_kwant = kwant_colorFit(img, palette)
        plt.subplot(2,2,2)
        plt.imshow(cv2.cvtColor(img_kwant, cv2.COLOR_BGR2RGB))
        plt.title('Kwantyzacja')
        plt.axis('off')

        img_dith_org = organized_dithering(img, palette, M2, 2)
        plt.subplot(2,2,3)
        plt.imshow(cv2.cvtColor(img_dith_org, cv2.COLOR_BGR2RGB))
        plt.title('Dithering Zorganizowany')
        plt.axis('off')

        img_fs = Floyd_Steinberg_dithering(img, palette)
        plt.subplot(2,2,4)
        plt.imshow(cv2.cvtColor(img_fs, cv2.COLOR_BGR2RGB))
        plt.title('Dithering Floyda-Steinberga')
        plt.axis('off')

        plt.show()

palletes2 = {
    "Palette 8": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]),
    "Palette 16": np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.75, 0.75, 0.75],
        [0.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
}
for name, img in small_images.items():
    for palette_name, palette in palletes2.items():
        plt.suptitle(f"{name} - {palette_name}")
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Oryginał')
        plt.axis('off')

        img_kwant = kwant_colorFit(img, palette)
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(img_kwant, cv2.COLOR_BGR2RGB))
        plt.title('Kwantyzacja')
        plt.axis('off')

        img_dith_org = organized_dithering(img, palette, M2, 2)
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img_dith_org, cv2.COLOR_BGR2RGB))
        plt.title('Dithering Zorganizowany')
        plt.axis('off')

        img_fs = Floyd_Steinberg_dithering(img, palette)
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(img_fs, cv2.COLOR_BGR2RGB))
        plt.title('Dithering Floyda-Steinberga')
        plt.axis('off')

        plt.show()