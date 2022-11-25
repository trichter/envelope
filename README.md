## Calculate spectral energy density or instantanious amplitude of seismic data (aka envelope)

Functions in the `envelope.py` module were copied from [Qopen](https://github.com/trichter/qopen).

```py
from obspy import read
from envelope import envelope, smooth_trace, spectral_energy_density

stream = read()
print(stream)
energy = spectral_energy_density(stream.copy(), 2, 10)
smoothed_energy = smooth_trace(energy.copy(), 0.2)
stream.filter('bandpass', freqmin=2, freqmax=10)
env = envelope(stream.copy())
print('envelope (spectral energy density):', energy)
print('smoothed envelope (spectral energy density):', smoothed_energy)
print('envelope (amplitude):', env)
stream.plot(handle=True)
env.plot(handle=True)
energy.plot(handle=True)
smoothed_energy.plot()
```
