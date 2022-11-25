# Copyright 2022 Tom Eulenfeld, MIT license
import numpy as np
import scipy.signal
import obspy


def filter_width(sr, freq=None, freqmin=None, freqmax=None, corners=2,
                 zerophase=False, type='bandpass'):
    """
    Integrate over the squared filter response of a Butterworth filter

    The result corresponds to the filter width, which equals approximately
    the difference of the corner frequencies. The energy density should
    be divided by the result to get the correct spectral energy density.

    :param sr: sampling rate
    :param freq: corner frequencies of low- or highpass filter
    :param freqmin,freqmax: corner frequencies of bandpass filter
    :param corners: number of corners
    :param zerophase: if True number of corners are doubled
    :param type: 'bandpass', 'highpass' or 'lowpass'
    :return: filter width
    """
    if type == 'bandpass':
        fs = (freqmin / (0.5 * sr), freqmax / (0.5 * sr))
    else:
        fs = freq / (0.5 * sr)
    b, a = scipy.signal.iirfilter(corners, fs, btype=type.strip('pass'),
                                  ftype='butter', output='ba')
    w, h = scipy.signal.freqz(b, a)
    df = (w[1] - w[0]) / 2 / np.pi * sr
    ret = df * np.sum(np.abs(h) ** (2 * (zerophase + 1)))
    return ret


def energy1c(data, rho, df, fs=4):
    """
    Spectral energy density of one channel

    :param data: velocity data (m/s)
    :param rho: density (kg/m**3)
    :param df: filter width in Hz
    :param fs: free surface correction (default: 4)
    :return: energy density
    """
    hilb = scipy.fftpack.hilbert(data)
    return rho * (data ** 2 + hilb ** 2) / 2 / df / fs


def analytic_signal(data):
    """
    Analytic (complex) signal
    """
    hilb = scipy.fftpack.hilbert(data)
    return (data ** 2 + hilb ** 2) ** 0.5


def envelope(stream):
    """
    Calculate envelope for each trace separately
    """
    for tr in stream:
        tr.data = np.abs(analytic_signal(tr.data))
    return stream


def total_energy(stream, rho=6000, df=1, fs=4, tolerance=1):
    """
    Return trace with total spectral energy density of three component stream

    :param stream: stream of a 3 component seismogram
    :param rho: density (kg/m**3)
    :param df: filter width in Hz
    :param fs: free surface correction (default: 4)
    :param tolerance: the number of samples the length of the traces
        in the 3 component stream may differ (default: 1)
    :return: trace with total energy density,
        units: if filtered, J/m**3/Hz, if unfiltered and df=1, J/m**3
    """
    data = [energy1c(tr.data, rho, df, fs=fs) for tr in stream]
    Ns = [len(d) for d in data]
    if max(Ns) - min(Ns) > tolerance:
        msg = ('traces for one stream have different lengths %s. Tolerance '
               ' is %d samples') % (Ns, tolerance)
        raise ValueError(msg)
    elif max(Ns) - min(Ns) > 0:
        data = [d[:min(Ns)] for d in data]
    data = np.sum(data, axis=0)
    tr = obspy.Trace(data=data, header=stream[0].stats)
    tr.stats.channel = tr.stats.channel[:2] + 'X'
    return tr


def spectral_energy_density(stream, freqmin, freqmax, corners=2,
                            zerophase=False, fs=4, rho=6000):
    """
    Return spectral energy density in specified frequency band

    Units: J/m**3/Hz
    """
    assert len(stream) == 3
    sr = stream[0].stats.sampling_rate
    assert (freqmin + freqmax) < sr
    if freqmax > 0.495 * sr:
        fu = {'freq': freqmin, 'type': 'highpass', 'zerophase': zerophase,
              'corners': corners}
    else:
        fu = {'freqmin': freqmin, 'freqmax': freqmax, 'type': 'bandpass',
              'zerophase': zerophase, 'corners': corners}
    stream.detrend('linear')
    stream.filter(**fu)
    df = filter_width(sr, **fu)
    return total_energy(stream, rho, df, fs=fs)


def smooth(x, window_len, window='flat', method='zeros'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects 'zeros', 'reflect', None
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))
        'reflect': pad reflected signal on both ends (same)
        None: no handling of border effects
            (len(smooth(x)) = len(x) - len(window_len) + 1)

    See also:
    www.scipy.org/Cookbook/SignalSmooth
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 2:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    else:
        s = x
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    return np.convolve(w / w.sum(), s, mode='valid')


def smooth_trace(tr, len_s, *args, **kwargs):
    """
    Smooth trace, specify window length in seconds
    """
    window_len = int(round(len_s * tr.stats.sampling_rate))
    tr.data = smooth(tr.data, window_len, *args, **kwargs)
    return tr


def smooth_stream(stream, *args, **kwargs):
    """
    Smooth stream, specify window length in seconds
    """
    for tr in stream:
        smooth_trace(tr, *args, **kwargs)
    return stream


if __name__ == '__main__':
    from obspy import read
    stream = read()
    print(stream)
    energy = spectral_energy_density(stream.copy(), 2, 10)
    smoothed_energy = smooth_trace(energy.copy(), 0.2)
    stream.filter('bandpass', freqmin=2, freqmax=10)
    env = envelope(stream.copy())
    print('envelope (amplitude):', env)
    print('envelope (energy density):', energy)
    print('smoothed envelope (energy density):', smoothed_energy)
    stream.plot(handle=True)
    env.plot(handle=True)
    energy.plot(handle=True)
    smoothed_energy.plot()
