import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import scipy.fftpack
import scipy.fft
from scipy.interpolate import interp1d


def plotAudio(signal, fs, time_margin=[0, 0.04], fsize=2 ** 10):
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, signal.shape[0]) / fs, signal)
    if len(time_margin) == 2:
        plt.xlim(time_margin)
    else:
        raise ValueError("time_margin must contain exactly two values")
    plt.title("Fragment sygnału w czasie")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")

    plt.subplot(2, 1, 2)
    plt.title("Widmo")
    yf = scipy.fftpack.fft(signal, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [dB]")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def saveAudio(data, fs, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    sf.write(path, data, fs)


def Kwant2(data, bits):
    d = 2 ** bits - 1
    if np.issubdtype(data.dtype, np.floating):
        data_clipped = np.clip(data, 0.0, 1.0)
        data_quantized = np.round(data_clipped * d) / d
    else:
        data_min = np.min(data).astype(np.float64)
        data_max = np.max(data).astype(np.float64)
        scale = d / (data_max - data_min) if data_max != data_min else 1
        dataF = data.astype(np.float64)
        data_quantized = np.round((dataF - data_min) * scale) / scale + data_min
        data_quantized = np.clip(data_quantized, data_min, data_max)
    return data_quantized.astype(data.dtype)

def decimation(data, n):
    return data[::n], data.shape[0] // n

def interpolation(data, fs, N, N1, kind='cubic'):
    t = np.linspace(0, N/fs, N, endpoint=False)
    t1 = np.linspace(0, N/fs, N1, endpoint=False)
    f = interp1d(t, data, kind=kind, fill_value="extrapolate")
    data1 = f(t1)
    return data1



# Zadanie 1
bit_depths = [4, 8, 16, 24]
decimations = [2, 4, 6, 10, 24]
interp_rates = [2000, 4000, 8000, 11999, 16000, 16953, 24000, 41000]
interpolation_kinds = ['linear', 'cubic']
file_names = ['SIN/sin_60Hz.wav', 'SIN/sin_440Hz.wav', 'SIN/sin_8000Hz.wav', 'SIN/sin_combined.wav']
for fname in file_names:
    if fname == 'SIN/sin_combined.wav':
        time_margin = [0, 0.008]
    elif fname == 'SIN/sin_8000Hz.wav':
        time_margin = [0, 0.0004]
    elif fname == 'SIN/sin_440Hz.wav':
        time_margin = [0, 0.006]
    else:
        time_margin = [0, 0.05]
    plt.suptitle(f"{fname}")
    data, fs = sf.read(fname, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    # 1. Kwantyzacja
    for bits in bit_depths:
        plt.suptitle(f"{fname} - Kwantyzacja: {bits} bit")
        quantized = Kwant2(data, bits)
        plotAudio(quantized, fs, time_margin)
    # 2. Decymacja
    for n in decimations:
        if fname == 'SIN/sin_8000Hz.wav':
            time_margin = [0, 0.004]
        plt.suptitle(f"{fname} - Decymacja: {n}")
        decimated, new_length = decimation(data, n)
        new_fs = fs // n
        plotAudio(decimated, new_fs, time_margin)
    # # 3. Interpolacja
    for rate in interp_rates:
        for kind in interpolation_kinds:
            if fname == 'SIN/sin_8000Hz.wav' and (rate == 2000 or rate == 4000 or rate == 8000):
                time_margin = [0, 1.0]
            elif fname == 'SIN/sin_8000Hz.wav' and (rate == 11999 or rate == 41000):
                time_margin = [0, 0.001]
            elif fname == 'SIN/sin_8000Hz.wav' and rate == 16000:
                time_margin = [0, 0.01]
            elif fname == 'SIN/sin_8000Hz.wav' and rate == 16953:
                time_margin = [0, 0.005]
            elif fname == 'SIN/sin_combined.wav' and rate == 24000:
                time_margin = [0, 0.0005]
            plt.suptitle(f"{fname} - Interpolacja: {rate}Hz - {kind}")
            N = len(data)
            interpolated = interpolation(data, fs, N, rate, kind=kind)
            plotAudio(interpolated, rate, time_margin)

# Zadanie 2
bit_depth = [4,8]
decimations =[2,4,6,10,24]
interp_rates = [4000,8000,11999,16000,16953]
file_names = ['SING/sing_low1.wav','SING/sing_medium1.wav','SING/sing_high1.wav']
interpolation_kinds = ['linear', 'cubic']
for fname in file_names:
    sanitized_fname = fname.replace(' ', '_').replace('/', '_')
    data, fs = sf.read(fname)
    if data.ndim > 1:
        data = data[:, 0]
    # 1. Kwantyzacja
    for bits in bit_depth:
        quantized = Kwant2(data, bits)
        saveAudio(quantized, fs, f"output/quantized", f"Kwant_{sanitized_fname}_{bits}bit.wav")
    # 2. Decymacja
    for n in decimations:
        decimated, new_length = decimation(data, n)
        new_fs = fs // n
        saveAudio(decimated, new_fs, f"output/decimated", f"Decimation_{sanitized_fname}_{n}bits.wav")
    # 3. Interpolacja
    for rate in interp_rates:
        for kind in interpolation_kinds:
            duration = len(data) / fs
            N1 = int(duration * rate)
            interp = interpolation(data, fs, len(data), N1, kind=kind)
            saveAudio(interp, rate, f"output/interpolated", f"Interp_{sanitized_fname}_{rate}Hz_{kind}.wav")