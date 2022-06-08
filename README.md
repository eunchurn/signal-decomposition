# Signal decomposition with Hankel matrix based SVD

[![README Generate](https://github.com/eunchurn/signal-decomposition/actions/workflows/readme-gen.yml/badge.svg)](https://github.com/eunchurn/signal-decomposition/actions/workflows/readme-gen.yml)

[**blog post**](https://www.eunchurn.com/blog/engineering/2018-01-20-signal-decomposition-hmbsvd)



```python
from scipy import signal, linalg
from scipy.sparse.linalg import svds, eigs
from scipy.linalg import hankel
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
```

### Read Audio File

- File: `lets_walk_vox_sample.wav` sampled by Lucid Fall's "걸어가자"


```python
samplerate, stereo_data = wavfile.read('./data/lets_walk_vox_sample.wav')
length = stereo_data.shape[0] / samplerate
print(stereo_data.shape[0])
time = np.linspace(0., length, stereo_data.shape[0])

fig1 = plt.figure(figsize=(14, 7), dpi=80)
plt.plot(time, stereo_data[:, 0], label="Left channel", linewidth=0.5)
plt.plot(time, stereo_data[:, 1], label="Right channel", linewidth=0.5)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
fig1.savefig("./figures/audioSample.pdf")
plt.show()
```

    95574



    
![png](main_files/main_3_1.png)
    


- Select only left channel


```python

data = stereo_data[:, 0]
```

### Signal decomposition

- 특이치분해에서 최대 100개의 반복수로 산정함
- 위의 100개의 rank가 나올 가능성이 높음.
- 이유는 tolerance 를 `1e-10`으로 설정했지만 데이터가 긴 경우 수렴하지 않을 가능성이 높음

```python
u, s, vt = svds(Hdata, k=nk, tol=1e-10, which="LM", maxiter=100)
```


```python
Hdata = hankel(data[0:2048], data[2048:-1])
nk = 100
u, s, vt = svds(Hdata, k=nk, tol=1e-10, which="LM", maxiter=100)
Fs = samplerate
NFFT = 2**14
P = np.asarray([0] * (len(data) - 1))  # 4095
P = P[:, np.newaxis]

```

- Build matrix of all signals
- Fourier transform with all decomposed signals


```python

Pxx = np.asarray([0] * 8193)
Pxx = Pxx[:, np.newaxis]
Pfft = np.asarray([0] * NFFT)
Pfft = Pfft[:, np.newaxis]
for x in range(0, nk - 1):
    v1 = vt[x, :]
    u1 = u[:, x]
    u1 = u1[:, np.newaxis]
    v1t = v1[np.newaxis, :]
    A1 = u1 * s[x] * v1t
    P1 = np.append(A1[0, :], A1[:, -1])
    # P1=A1[0,:]
    P = np.hstack((P, P1[:, np.newaxis]))
    F, Pxx_den = signal.welch(P1, fs=Fs, window="hamm", nfft=NFFT, detrend="constant")
    ft = np.fft.fft(P1, NFFT)
    Pxx = np.hstack((Pxx, Pxx_den[:, np.newaxis]))
    Pfft = np.hstack((Pfft, ft[:, np.newaxis]))
```

- `P`가 분리된 신호들의 집합(Matrix) 이다.

### Plotting all decomposed signals

- Time domain of all decomposed signals


```python
fig2 = plt.figure(figsize=(14, 7), dpi=80)
plt.plot(P, linewidth=0.2)
plt.xlabel("Sample")
plt.ylabel("Amplitude")
fig2.savefig("./figures/timeDomain.pdf")
plt.show()
```


    
![png](main_files/main_12_0.png)
    


- Frequency domain of all decomposed signals


```python
freq = np.fft.fftfreq(NFFT,1/Fs)
Nyfreq = freq[: int(NFFT / 2 - 1)]
extent = [Nyfreq[0], Nyfreq[-1], 0, nk - 1]

fig3 = plt.figure(figsize=(14, 7), dpi=80)
plt.semilogy(Nyfreq, np.abs(Pfft[: int(NFFT / 2 - 1)]), linewidth=0.1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
fig3.savefig("./figures/allFFTs.pdf")
plt.show()
```


    
![png](main_files/main_14_0.png)
    


- Subplot 1: Spectrogram of all decomposed signal
- Subplot 2: Sum of all decomposed FFT data


```python
fig4, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), dpi=80, sharex=True)
ax1.imshow(
    np.log10(np.abs(Pfft[: int(NFFT / 2 - 1), :]).transpose()),
    aspect="auto",
    interpolation="none",
    extent=extent,
    cmap=plt.cm.plasma,
    origin="lower",
)
ax1.set_xlim([0, Fs / 2])
ax1.set_ylim([1, nk])
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Rank")

ax2.semilogy(Nyfreq, np.abs(Pfft[: int(NFFT / 2 - 1)].sum(axis=1)), "k", linewidth=0.1)
ax2.set_xlim([0, Fs / 2])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Sum of FFTs")
fig4.savefig("./figures/spectrogram.pdf")
plt.show()
```

    /tmp/ipykernel_1630/1271570416.py:3: RuntimeWarning: divide by zero encountered in log10
      np.log10(np.abs(Pfft[: int(NFFT / 2 - 1), :]).transpose()),



    
![png](main_files/main_16_1.png)
    


- 0~3kHz frequency zoomed


```python
fig5, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), dpi=80, sharex=True)
ax1.imshow(
    np.log10(np.abs(Pfft[: int(NFFT / 2 - 1), :]).transpose()),
    aspect="auto",
    interpolation="none",
    extent=extent,
    cmap=plt.cm.plasma,
    origin="lower",
)
ax1.set_xlim([0, Fs / 16])
ax1.set_ylim([1, nk])
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Rank")

ax2.semilogy(Nyfreq, np.abs(Pfft[: int(NFFT / 2 - 1)].sum(axis=1)), "k", linewidth=0.1)
ax2.set_xlim([0, Fs / 16])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Sum of FFTs")
fig5.savefig("./figures/spectrogramZoomed.pdf")
plt.show()
```

    /tmp/ipykernel_1630/3813461710.py:3: RuntimeWarning: divide by zero encountered in log10
      np.log10(np.abs(Pfft[: int(NFFT / 2 - 1), :]).transpose()),



    
![png](main_files/main_18_1.png)
    

