import numpy as np
from typing import Literal
from scipy.signal import butter, filtfilt
from scipy.fft import ifft

class Signal:
    def __init__(self, *args):
        self._name = args['name']

        self.s_min = args['signal_min']
        self.s_max = args['signal_max']
        self.t_min = args['t_min']
        self.t_max = args['t_max']

        # setup for filtering
        self.cutoff_freq = args["cutoff_freq"]
        self.btype = args["btype"]
        self.order = args["order"]
        
        # setup for DFT
        if args['sr'] == None:
            self.sr = 20 # 20Hz
        else:
            self.sr = args["sr"]

        self.s_arr = self.filter(args['signal'], self.cutoff_freq, self.sr, self.order, self.btype)
        
    def __call__(self, vtype:Literal['Original', 'Fourier'] = "Original"):
        if vtype == 'Original':
            return self.s_arr
        else:
            return self.transform(self.s_arr, self.sr)

    def filter(self, signal: np.array, cutoff_freq: int, fs: int, order: int, btype:Literal['highpass', 'losspass']):
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
        filtered = filtfilt(b, a, signal)
        return filtered

    def _DFT(self, x:np.array):
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        X = np.dot(e, x)
        return X

    def transform(self, signal:np.array, sr:int):
        dft = self._DFT(signal)
        N = len(dft)
        n = np.arange(N)
        T = N/sr
        freq = n/T 
        n_oneside = N // 2
        freq_oneside = freq[:n_oneside]
        dft_oneside = dft[:n_oneside]
        return freq_oneside, dft_oneside

    def inverse_transform(self, dft:np.array):
        n = int(self.sr * (self.t_max - self.t_min))
        signal_recovery = ifft(dft, n)
        return signal_recovery

    def get_actuator_name(self):
        return self._name