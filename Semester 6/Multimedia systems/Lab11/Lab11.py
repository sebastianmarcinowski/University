import cv2
import numpy as np
import matplotlib.pyplot as plt
from networkx.classes.filters import hide_edges, hide_nodes
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image

# Funkcja do znakowania wodnego
def water_mark(img,mask,alpha=0.25):
    assert (img.shape[0]==mask.shape[0]) and (img.shape[1]==mask.shape[1]), "Wrong size"
    if len(img.shape)<3:
        flag=True
        t_img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)
    else:
        flag=False
        t_img=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    if (mask.dtype==bool):
        t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
    elif (mask.dtype==np.uint8):
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    else:
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    t_out=cv2.addWeighted(t_img,1,t_mask,alpha,0)
    if flag:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2GRAY)
    else:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2RGB)
    return out


def put_data(img, data, binary_mask=np.uint8(1)):
    assert img.dtype == np.uint8, "img wrong data type"
    assert binary_mask.dtype == np.uint8, "binary_mask wrong data type"

    un_binary_mask = np.unpackbits(binary_mask)
    if data.dtype != bool:
        unpacked_data = np.unpackbits(data)
    else:
        unpacked_data = data

    dataspace = img.shape[0] * img.shape[1] * np.sum(un_binary_mask)

    if dataspace < unpacked_data.size:
        assert (dataspace >= unpacked_data.size), "too much data"
    if dataspace == unpacked_data.size:
        prepared_data = unpacked_data.reshape(img.shape[0], img.shape[1], np.sum(un_binary_mask)).astype(np.uint8)
    else:
        padded_data = np.zeros(dataspace, dtype=np.uint8)
        padded_data[:unpacked_data.size] = unpacked_data
        prepared_data = padded_data.reshape(img.shape[0], img.shape[1], np.sum(un_binary_mask)).astype(np.uint8)

    mask = np.full((img.shape[0], img.shape[1]), binary_mask)
    img = np.bitwise_and(img, np.invert(mask))
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            temp = prepared_data[:, :, bv]
            temp = np.left_shift(temp, i)
            img = np.bitwise_or(img, temp)
            bv += 1
    return img


def pop_data(img, binary_mask=np.uint8(1), out_shape=None):
    un_binary_mask = np.unpackbits(binary_mask)
    data = np.zeros((img.shape[0], img.shape[1], np.sum(un_binary_mask))).astype(np.uint8)
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            mask = np.full((img.shape[0], img.shape[1]), 2 ** i)
            temp = np.bitwise_and(img, mask)
            temp = np.right_shift(temp, i)  # POPRAWKA: dodano przesunięcie w prawo
            data[:, :, bv] = temp[:, :].astype(np.uint8)
            bv += 1
    if out_shape != None:
        tmp = np.packbits(data.flatten())
        tmp = tmp[:np.prod(out_shape)]
        data = tmp.reshape(out_shape)
    return data

# # Funckja kodująca informacje w 1 warstwie obrazu
# def put_data(img,data,binary_mask=np.uint8(1)):
#     assert img.dtype==np.uint8 , "img wrong data type"
#     assert binary_mask.dtype==np.uint8, "binary_mask wrong data type"
#     un_binary_mask=np.unpackbits(binary_mask)
#     if data.dtype!=bool:
#         unpacked_data=np.unpackbits(data)
#     else:
#         unpacked_data=data
#     dataspace=img.shape[0]*img.shape[1]*np.sum(un_binary_mask)
#     assert (dataspace>=unpacked_data.size) , "too much data"
#
#     total_bits_needed = img.shape[0] * img.shape[1] * np.sum(un_binary_mask)
#     if unpacked_data.size < total_bits_needed:
#         padded_data = np.zeros(total_bits_needed, dtype=np.uint8)
#         padded_data[:unpacked_data.size] = unpacked_data
#         unpacked_data = padded_data
#     else:
#         unpacked_data = unpacked_data[:total_bits_needed]
#
#     prepered_data = unpacked_data.reshape(img.shape[0], img.shape[1], np.sum(un_binary_mask)).astype(np.uint8)
#     mask=np.full((img.shape[0],img.shape[1]),binary_mask)
#     img=np.bitwise_and(img,np.invert(mask))
#     bv=0
#     for i,b in enumerate(un_binary_mask[::-1]):
#         if b:
#             temp=prepered_data[:,:,bv]
#             temp=np.left_shift(temp,i)
#             img=np.bitwise_or(img,temp)
#             bv+=1
#     return img
#
# def pop_data(img,binary_mask=np.uint8(1),out_shape=None):
#     un_binary_mask=np.unpackbits(binary_mask)
#     data=np.zeros((img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
#     bv=0
#     for i,b in enumerate(un_binary_mask[::-1]):
#         if b:
#             mask=np.full((img.shape[0],img.shape[1]),2**i)
#             temp=np.bitwise_and(img,mask)
#             data[:,:,bv]=temp[:,:].astype(np.uint8)
#             bv+=1
#     if out_shape!=None:
#         tmp=np.packbits(data.flatten())
#         tmp=tmp[:np.prod(out_shape)]
#         data=tmp.reshape(out_shape)
#     return data


# Zadanie 2
def zad2():
    img = cv2.imread('color1.jpg')
    blue_channel = img[:, :, 0]
    with open("tekst.txt",'r') as file:
        text = file.read()
    mask = np.uint8(1)
    text_data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    encoded_text = put_data(blue_channel, text_data, mask)
    encoded_img = img.copy()
    encoded_img[:, :, 0] = encoded_text
    decoded_bits = pop_data(encoded_text, mask, out_shape=text_data.shape)
    decoded_text = decoded_bits.tobytes().decode('utf-8', errors='ignore')

    psnr_before = psnr(blue_channel, encoded_text)
    ssim_before = ssim(blue_channel, encoded_text, data_range=encoded_text.max() - encoded_text.min())
    psnr_after = psnr(encoded_text, blue_channel)
    ssim_after = ssim(blue_channel, blue_channel, data_range=blue_channel.max() - blue_channel.min())

    print(f"Pierwotny tekst: {text}")
    print(f"Odtworzony tekst: {decoded_text}")
    print(f"PSNR przed: {psnr_before:.2f}")
    print(f"SSIM przed: {ssim_before:.4f}")
    print(f"PSNR po: {psnr_after:.2f}")
    print(f"SSIM po: {ssim_after:.4f}")
# zad2()

# Zadanie 3
def zad3():
    carrier_img = cv2.imread('color1.jpg')
    carrier_img = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2RGB)
    carrier_img_orig = carrier_img.copy()
    hide_img = cv2.imread('color2.jpg')
    hide_img = cv2.cvtColor(hide_img, cv2.COLOR_BGR2RGB)

    scale_factor = 0.5
    while True:
        hide_img_res = cv2.resize(hide_img, (int(carrier_img.shape[1] * scale_factor), int(carrier_img.shape[0] * scale_factor)))
        hide_b, hide_g, hide_r = cv2.split(hide_img_res)
        carrier_b, carrier_g, carrier_r = cv2.split(carrier_img)
        dataspace_b = carrier_b.size * 3
        dataspace_g = carrier_g.size * 2
        dataspace_r = carrier_g.size * 2
        if hide_b.size * 8 <= dataspace_b and hide_g.size * 8 <= dataspace_g and hide_r.size * 8 <= dataspace_r:
            break
        scale_factor *= 0.9
    hide_img = hide_img_res
    hide_b, hide_g, hide_r = cv2.split(hide_img)

    binary_mask_b = np.uint8(7)
    binary_mask_r = np.uint8(3)
    binary_mask_g = np.uint8(3)

    encoded_b = put_data(carrier_b.copy(), hide_b, binary_mask_b)
    encoded_g = put_data(carrier_g.copy(), hide_g, binary_mask_g)
    encoded_r = put_data(carrier_r.copy(), hide_r, binary_mask_r)

    recovered_b = pop_data(encoded_b, binary_mask_b, out_shape=hide_b.shape)
    recovered_g = pop_data(encoded_g, binary_mask_g, out_shape=hide_g.shape)
    recovered_r = pop_data(encoded_r, binary_mask_r, out_shape=hide_r.shape)

    # Scalone obrazy
    encoded_img = cv2.merge((encoded_b, encoded_g, encoded_r))
    recovered_img = cv2.merge((recovered_b, recovered_g, recovered_r))


    original_b, original_g, original_r = cv2.split(carrier_img_orig)

    # PSNR i SSIM
    psnr_carrier_b = psnr(original_b, encoded_b)
    psnr_carrier_g = psnr(original_g, encoded_g)
    psnr_carrier_r = psnr(original_r, encoded_r)
    ssim_carrier_b = ssim(original_b, encoded_b, data_range=255)
    ssim_carrier_g = ssim(original_g, encoded_g, data_range=255)
    ssim_carrier_r = ssim(original_r, encoded_r, data_range=255)
    print(f"PSNR nosiciela (B): {psnr_carrier_b:.2f}, SSIM nosiciela (B): {ssim_carrier_b:.4f}")
    print(f"PSNR nosiciela (G): {psnr_carrier_g:.2f}, SSIM nosiciela (G): {ssim_carrier_g:.4f}")
    print(f"PSNR nosiciela (R): {psnr_carrier_r:.2f}, SSIM nosiciela (R): {ssim_carrier_r:.4f}")

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(carrier_img_orig)
    plt.title('Oryginalny nosiciel')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(encoded_img)
    plt.title('Obraz zakodowany')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(hide_img)
    plt.title('Ukryty obraz (oryginalny)')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(recovered_img)
    plt.title('Obraz odzyskany')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
zad3()

# Zadanie 4
def zad4():
    carrier_img = cv2.imread('color2.jpg')
    carrier_img = cv2.cvtColor(carrier_img, cv2.COLOR_BGR2RGB)
    carrier_b, carrier_g, carrier_r = cv2.split(carrier_img)
    img_height, img_width = carrier_img.shape[0], carrier_img.shape[1]
    bit_configs = [
        ("0 bits (no data)", 0, 0, 0, 0),
        ("1 bit (B only)", 0, 0, 1, 1),
        ("2 bits (RB)", 1, 0, 1, 2),
        ("3 bits (RGB)", 1, 1, 1, 3),
        ("4 bits (2R,2B)", 2, 0, 2, 4),
        ("5 bits (2R,1G,2B)", 2, 1, 2, 5),
        ("6 bits (2RGB)", 2, 2, 2, 6),
        ("7 bits (3R,2G,2B)", 3, 2, 2, 7),
        ("8 bits (3R,2G,3B)", 3, 2, 3, 8),
        ("9 bits (3RGB)", 3, 3, 3, 9),
        ("10 bits (4R,3G,3B)", 4, 3, 3, 10),
        ("11 bits (4R,4G,3B)", 4, 4, 3, 11),
        ("12 bits (4RGB)", 4, 4, 4, 12),
        ("13 bits (5R,4G,4B)", 5, 4, 4, 13),
        ("14 bits (5R,5G,4B)", 5, 5, 4, 14),
        ("15 bits (5RGB)", 5, 5, 5, 15),
        ("16 bits (6R,5G,5B)", 6, 5, 5, 16),
        ("17 bits (6R,6G,5B)", 6, 6, 5, 17),
        ("18 bits (6RGB)", 6, 6, 6, 18),
        ("19 bits (7R,6G,6B)", 7, 6, 6, 19),
        ("20 bits (7R,7G,6B)", 7, 7, 6, 20),
        ("21 bits (7RGB)", 7, 7, 7, 21),
        ("22 bits (8R,7G,7B)", 8, 7, 7, 22),
        ("23 bits (8R,8G,7B)", 8, 8, 7, 23),
        ("24 bits (8RGB)", 8, 8, 8, 24),
    ]

    for desc, r_bits, g_bits, b_bits, total_bits in bit_configs:
        r_mask = np.uint8((2 ** r_bits) - 1) if r_bits > 0 else np.uint8(0)
        g_mask = np.uint8((2 ** g_bits) - 1) if g_bits > 0 else np.uint8(0)
        b_mask = np.uint8((2 ** b_bits) - 1) if b_bits > 0 else np.uint8(0)

        r_capacity_bits = img_height * img_width * r_bits
        g_capacity_bits = img_height * img_width * g_bits
        b_capacity_bits = img_height * img_width * b_bits

        random_data_r_bytes = np.ceil(r_capacity_bits / 8).astype(int)
        random_data_g_bytes = np.ceil(g_capacity_bits / 8).astype(int)
        random_data_b_bytes = np.ceil(b_capacity_bits / 8).astype(int)

        data_r = np.random.randint(0, 256, size=random_data_r_bytes, dtype=np.uint8) if r_bits > 0 else np.array([])
        data_g = np.random.randint(0, 256, size=random_data_g_bytes, dtype=np.uint8) if g_bits > 0 else np.array([])
        data_b = np.random.randint(0, 256, size=random_data_b_bytes, dtype=np.uint8) if b_bits > 0 else np.array([])

        encoded_r = put_data(carrier_r, data_r, r_mask) if r_bits > 0 else carrier_r
        encoded_g = put_data(carrier_g, data_g, g_mask) if g_bits > 0 else carrier_g
        encoded_b = put_data(carrier_b, data_b, b_mask) if b_bits > 0 else carrier_b

        encoded_img = cv2.merge((encoded_r, encoded_g, encoded_b))

        psnr_val = psnr(carrier_img, encoded_img)
        ssim_val = ssim(carrier_img, encoded_img, win_size=3, channel_axis=-1)

        print(f"PSNR nosiciela: {psnr_val:.2f}")
        print(f"SSIM nosiciela: {ssim_val:.4f}")

        fix, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs[0].imshow(carrier_img)
        axs[0].set_title("Obraz nosiciel")
        encoded_img = cv2.cvtColor(encoded_img, cv2.COLOR_RGB2BGR)
        axs[1].imshow(encoded_img)
        axs[1].set_title(f"Obraz zakodowany: {desc}")
        plt.tight_layout()
        plt.show()
zad4()

# Zadanie 5
def zad5():
    img = cv2.imread('color1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread('binary1.jpg', cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))  # dopasowanie rozmiaru

    for alpha in [0.10, 0.25, 0.50]:
        watermarked_img = water_mark(img_rgb, mask_resized, alpha)

        # Obliczenia PSNR/SSIM
        psnr_val = psnr(img_rgb, watermarked_img)
        ssim_val = ssim(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2GRAY),
                        data_range=255)

        print(f"Alpha = {alpha:.2f} => PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

        # Wyniki
        plt.figure(figsize=(10, 5))
        plt.title(f"Znak wodny (alpha={alpha})")
        plt.imshow(watermarked_img)
        plt.axis('off')
        plt.tight_layout()
        # plt.savefig(f"watermarked_alpha_{int(alpha * 100)}.png")
        plt.show()
# zad5()