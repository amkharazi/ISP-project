import numpy as np

def apply_band_filters(freq_data, sample_rate, band_ranges):
    '''
    Apply bandpass filters to frequency domain data.

    Parameters:
        - freq_data : The frequency domain data with shape (..., D, C).
        - sample_rate : The sampling rate of the original signal.
        - band_ranges : A list of frequency band ranges, e.g., [(low1, high1), (low2, high2)].

    Returns:
        - combined_filtered: Filtered frequency data with relevant frequencies combined.
    '''
    freq_bins = np.fft.fftfreq(freq_data.shape[-2], d=1/sample_rate)
    combined_filtered = np.zeros_like(freq_data, dtype=complex)

    for low, high in band_ranges:
        band_mask = (freq_bins >= low) & (freq_bins <= high)
        combined_filtered[..., band_mask, :] += freq_data[..., band_mask, :]

    return combined_filtered

def frequency_domain_processing(X, sample_rate, time = False):
    '''
    Convert signals to frequency domain and apply typical bandpass filters.

    Parameters:
        - X : Input array with shape (..., D, C), where D is the time dimension and C is the channel dimension.
        - sample_rate : The sampling rate of the input signals.
        - time : whethers the inverted fft or not. default it False. 

    Returns:
        - filtered_freq_data: Signals processed in the frequency domain with typical bandpass filters applied.
    '''
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    freq_data = np.fft.fft(X, axis=-2)

    filtered_freq_data = apply_band_filters(freq_data, sample_rate, list(band_ranges.values()))

    if time:
        filtered_time_data = np.fft.ifft(filtered_freq_data, axis=-2).real
        return filtered_time_data

    return filtered_freq_data



def preprocess(X, extra_info, strategy = None):


    if strategy=='freq':
        X = frequency_domain_processing(X=X, sample_rate=extra_info['srate'], time=False)
    elif strategy=='freq-time':
        X = frequency_domain_processing(X=X, sample_rate=extra_info['srate'], time=True)

        



    


    