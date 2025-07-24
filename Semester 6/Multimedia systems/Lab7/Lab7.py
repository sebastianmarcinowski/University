import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

A = 87.6
mu = 255
def Kwant(data, bits):
    d = 2 ** bits
    step = 2 / (d- 1)
    data_quantized = np.round((data + 1) / step) * step - 1
    return data_quantized

def A_LAW_encoder(signal):
    A = 87.6
    denominator = 1 + np.log(A)
    signal = signal.copy()
    indexes = np.abs(signal) < 1 / A
    signal[indexes] = np.sign(signal[indexes]) * (A * np.abs(signal[indexes]) / denominator)
    signal[~indexes] = np.sign(signal[~indexes]) * ((1 + np.log(A * np.abs(signal[~indexes]))) / denominator)
    return signal

def A_LAW_decoder(encoded_signal):
    A = 87.6
    denominator = 1 + np.log(A)
    encoded_signal = encoded_signal.copy()
    indexes = np.abs(encoded_signal) < 1 / denominator
    encoded_signal[indexes] = np.sign(encoded_signal[indexes]) * (np.abs(encoded_signal[indexes]) * denominator / A)
    encoded_signal[~indexes] = np.sign(encoded_signal[~indexes]) * (np.exp(np.abs(encoded_signal[~indexes]) * denominator - 1) / A)
    return encoded_signal

def mu_LAW_encoder(signal):
    mu = 255
    signal = signal.copy()
    indexes = (-1 <= signal) & (signal <= 1)
    signal[indexes] = np.sign(signal[indexes]) * (np.log(1 + mu * np.abs(signal[indexes])) / np.log(1 + mu))
    return signal

def mu_LAW_decoder(encoded_signal):
    mu = 255
    encoded_signal = encoded_signal.copy()
    indexes = (-1 <= encoded_signal) & (encoded_signal <= 1)
    encoded_signal[indexes] = np.sign(encoded_signal[indexes]) * (1 / mu) * (np.expm1(np.abs(encoded_signal[indexes]) * np.log(1 + mu)))
    return encoded_signal

def DPCM_encoder(signal, bit):
    y = np.zeros(signal.shape)
    e = 0
    for i in range(0, signal.shape[0]):
        y[i] = Kwant(signal[i] - e, bit)
        e += y[i]
    return y

def DPCM_decoder(encoded_signal):
    decoded_signal = np.zeros_like(encoded_signal)
    e = 0
    for i in range(len(encoded_signal)):
        decoded_signal[i] = e + encoded_signal[i]
        e = decoded_signal[i]
    return decoded_signal

def DPCM_encoder_prediction(signal, bit, n):
    encoded_signal = np.zeros(signal.shape)
    reconstructed_signal = np.zeros(signal.shape)
    e = 0
    for i in range(0, signal.shape[0]):
        encoded_signal[i] = Kwant(signal[i] - e, bit)
        reconstructed_signal[i] = encoded_signal[i] + e
        if i > 0:
            e = np.mean(reconstructed_signal[max(0, i - n):i])
        else:
            e = 0
    return encoded_signal

def DPCM_decoder_prediction(encoded_signal, n):
    decoded_signal = np.zeros_like(encoded_signal)
    e = 0
    for i in range(len(encoded_signal)):
        decoded_signal[i] = e + encoded_signal[i]
        if i >= n:
            e = np.mean(decoded_signal[i - n:i])
        else:
            e = np.mean(decoded_signal[:i + 1])
    return decoded_signal

def test1():
    x = np.linspace(-1,1,1000)
    x_kwant = Kwant(x, 8)

    y1 = A_LAW_encoder(x.copy())
    y1_kwant = Kwant(y1, 8)

    y2 = mu_LAW_encoder(x.copy())
    y2_kwant = Kwant(y2, 8)

    y1_decoded_a = A_LAW_decoder(y1.copy())
    y1_decoded_a_kwant = A_LAW_decoder(y1_kwant.copy())

    y2_decoded_mu = mu_LAW_decoder(y2.copy())
    y2_decoded_mu_kwant = mu_LAW_decoder(y2_kwant.copy())


    plt.figure(figsize=(10, 8))
    plt.plot(x, y1_kwant, label='Sygnal po kompresji a-law po kwantyzacji do 8-bitow')
    plt.plot(x, y1, label='Sygnal po kompresji a-law bez kwantyzacji')
    plt.plot(x, y2_kwant, label='Sygnal po kompresji mu-law po kwantyzacji do 8-bitow')
    plt.plot(x, y2, label='Sygnal po kompresji mu-law bez kwantyzacji')
    plt.title("Krzywa kompresji")
    plt.xlabel("Wartość sygnału wejściowego")
    plt.ylabel("Wartość sygnału wyjściowego")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(x, x, label='Sygnał oryginalny')
    plt.plot(x, y1_decoded_a_kwant, label='Sygnał po dekompresji z a-law (kwantyzacja 8-bitów)')
    plt.plot(x, y2_decoded_mu_kwant, label='Sygnał po dekompresji z mu-law (kwantyzacja 8-bitów)')
    plt.plot(x, x_kwant, label='Sygnał oryginalny po kwantyzacji do 8 bitów')
    plt.title("Krzywa dekompresji")
    plt.xlabel("Wartość sygnału wejściowego")
    plt.ylabel("Wartość sygnału wyjściowego")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def przyklad_A():
    x = np.linspace(-1, 1, 1000)
    y = 0.9 * np.sin(np.pi * x * 4)

    a_law_kwant = Kwant(A_LAW_encoder(y.copy()), 6)
    a_law_decoded = A_LAW_decoder(a_law_kwant.copy())
    mu_law_kwant = Kwant(mu_LAW_encoder(y.copy()), 6)
    mu_law_decoded = mu_LAW_decoder(mu_law_kwant.copy())
    dpcm = DPCM_encoder(y.copy(), 6)
    dpcm_decoded = DPCM_decoder(dpcm.copy())
    dpcm_pred = DPCM_encoder_prediction(y.copy(), 6, 3)
    dpcm_pred_decoded = DPCM_decoder_prediction(dpcm_pred.copy(), 3)

    fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Przykład A kwantyzacja do 6 bitów")
    axs[0].plot(x, y)
    axs[0].set_title("Oryginalny sygnal")
    axs[1].plot(x, a_law_decoded)
    axs[1].set_title("A-LAW")
    axs[2].plot(x, mu_law_decoded)
    axs[2].set_title("MU-LAW")
    axs[3].plot(x, dpcm_decoded)
    axs[3].set_title("DPCM")
    axs[4].plot(x, dpcm_pred_decoded)
    axs[4].set_title("DPCM predykcja")
    plt.show()

def przyklad_B():
    x = np.linspace(-1, 1, 1000)
    y = 0.9 * np.sin(np.pi * x * 4)

    a_law_kwant = Kwant(A_LAW_encoder(y.copy()), 6)
    a_law_decoded = A_LAW_decoder(a_law_kwant.copy())
    mu_law_kwant = Kwant(mu_LAW_encoder(y.copy()), 6)
    mu_law_decoded = mu_LAW_decoder(mu_law_kwant.copy())
    dpcm = DPCM_encoder(y.copy(), 6)
    dpcm_decoded = DPCM_decoder(dpcm.copy())
    dpcm_pred = DPCM_encoder_prediction(y.copy(), 6, 3)
    dpcm_pred_decoded = DPCM_decoder_prediction(dpcm_pred.copy(), 3)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Sygnał oryginalny')
    plt.plot(x, a_law_decoded, label='Sygnał po dekompresji z a-law')
    plt.plot(x, mu_law_decoded, label='Sygnał po dekompresji z mu-law')
    plt.plot(x, dpcm_decoded, label='Sygnał po dekompresji z DPCM')
    plt.plot(x, dpcm_pred_decoded, label='Sygnał po dekompresji z DPCM z predykcją')
    plt.title("Przykład B kwantyzacja do 6 bitów")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def saveAudio(data, fs, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    sf.write(path, data, fs)


def sound_files():
    bits = [8, 7, 6, 5, 4, 3, 2]
    # bits=[8]
    file_names = ['SING/sing_low1.wav', 'SING/sing_medium1.wav', 'SING/sing_high1.wav']
    for fname in file_names:
        for bit in bits:
            sanitized_fname = fname.replace(' ', '_').replace('/', '_')
            data, fs = sf.read(fname)
            if data.ndim > 1:
                data = data[:, 0]
            # A-LAW
            a_law_kwant = Kwant(A_LAW_encoder(data.copy()), bit)
            a_law_decoded = A_LAW_decoder(a_law_kwant.copy())
            saveAudio(a_law_decoded, fs, "output", f"{bit}bits_A_LAW_{sanitized_fname}")
            # MU-LAW
            mu_law_kwant = Kwant(mu_LAW_encoder(data.copy()), bit)
            mu_law_decoded = mu_LAW_decoder(mu_law_kwant.copy())
            saveAudio(mu_law_decoded, fs, "output", f"{bit}bits_MU_LAW_{sanitized_fname}")
            # DPCM
            dpcm_encoded = DPCM_encoder(data.copy(), bit)
            dpcm_decoded = DPCM_decoder(dpcm_encoded.copy())
            saveAudio(dpcm_decoded, fs, "output", f"{bit}bits_DPCM_{sanitized_fname}")
            # DPCM with prediction
            dpcm_pred_encoded = DPCM_encoder_prediction(data.copy(), bit, 10)
            dpcm_pred_decoded = DPCM_decoder_prediction(dpcm_pred_encoded.copy(), 10)
            saveAudio(dpcm_pred_decoded, fs, "output", f"{bit}bits_DPCM_Prediction_{sanitized_fname}")

# test1()
# przyklad_A()
# przyklad_B()
sound_files()