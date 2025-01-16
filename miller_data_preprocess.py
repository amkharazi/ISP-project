import numpy as np
import torch
import torch.nn as nn


def apply_band_filters(freq_data, sample_rate, band_ranges):
    '''
    Apply bandpass filters to the frequency-domain data.

    Parameters:
        - freq_data : Input array in the frequency domain with shape (..., D, C), 
            where D is the frequency dimension and C is the channel dimension.
        - sample_rate : Sampling rate of the original signal.
        - band_ranges : List of frequency band ranges, e.g., [(0.5, 4), (4, 8), ...].

    Returns:
        - filtered_data : Filtered frequency domain data with shape (..., N, D, C), where N is the number of bands.
            If N == 1, the output will have shape (..., D, C), squeezing the band dimension.
    '''
    D = freq_data.shape[-2]
    freqs = np.fft.fftfreq(D, d=1/sample_rate)
    
    band_filters = []
    for low, high in band_ranges:
        band_filter = (freqs >= low) & (freqs < high)
        band_filter |= (freqs <= -low) & (freqs > -high) 
        band_filters.append(band_filter)

    filtered_data = np.array([
        freq_data * band_filter[..., np.newaxis] for band_filter in band_filters
    ])

    filtered_data = np.moveaxis(filtered_data, 0, -3)

    if len(band_ranges) == 1:
        filtered_data = np.squeeze(filtered_data, axis=-3)
    
    return filtered_data

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

class AdjustableMeanPooling(nn.Module):
    '''
    Adjustable average pooling. Based on the kernel size, computes the averages accross the signals. Designed for the miller dataset and initiated dataset.
    '''
    def __init__(self, kernel_size):
        super(AdjustableMeanPooling, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)
    
    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        original_shape = x.shape
        
        *batch_dims, D, C = original_shape
        
        B = np.prod(batch_dims)
        x = x.view(B, 1, C, D)
        
        x = self.pool(x)        
        x = x.view(*batch_dims, x.shape[-1], x.shape[-2])  
        
        return x.numpy()
class AdjustableModePooling(nn.Module):
    '''
    Adjustable mode pooling. Based on the kernel size, computes the mode accross the signals. Designed for the miller dataset and initiated dataset.
    This is used to adjust the sizes of the ouput tensor (target Y) when adjustable average pooling is used.
    '''
    def __init__(self, kernel_size):
        super(AdjustableModePooling, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        original_shape = x.shape
        *batch_dims, D, C = original_shape
        
        B = np.prod(batch_dims)
        x = x.view(B, 1, C, D)
        
        pooled_results = []
        
        for i in range(0, C - self.kernel_size[0] + 1, self.kernel_size[0]):  
            pool = []
            for j in range(0, D - self.kernel_size[1] + 1, self.kernel_size[1]):  
                window = x[:, :, i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                
                mode_values, _ = torch.mode(window.view(-1, window.shape[-1]), dim=1)
                pool.append(mode_values)
            pooled_results.append(torch.stack(pool, dim=0))

        pooled_results = torch.stack(pooled_results, dim=0)
        pooled_results = pooled_results.view(*batch_dims, pooled_results.shape[1], pooled_results.shape[0])
        
        return pooled_results.numpy()

def threshold_eeg_data(X, threshold, keep_original = False):
    '''
    Converts ECoG data into a neural-like binary format with a given threshold.
    
    Args:
    - X : Input ECoG data with shape (..., D, C),
                                  where D is timepoints and C is channels.
    - threshold : The threshold value. Values above this threshold are set to 1, otherwise to 0.

    - keep_original : Defines whether to return the with original values or just binary format. By default it returns only the binary format data.
    
    Returns:
    - binary_data: ECoG data in a binary format, same shape as input.
    '''
    binary_data = np.where(X > threshold, 1, 0)

    if keep_original:
        return X*binary_data
    
    return binary_data

def preprocess(X, Y, extra_info, strategy=None, init_strategy='simple', kernel_size=(1, 10), threshold=0.65):
    '''
    Preprocesses the input data (X and Y) based on the specified strategy. The function applies different 
    transformations to the data to prepare it for further analysis or model training. 
    
    Args:
    - X : Input data, typically ECoG signals, with shape (..., D, C),
                                          where D is timepoints and C is channels.
    - Y : Labels or target data, corresponding to X. The shape is typically
                                          aligned with X, but may vary depending on the context.
    - extra_info : A dictionary containing additional information such as the sampling rate (srate).
    - strategy : The preprocessing strategy to apply. Options are:
        - 'freq': Apply frequency domain processing without time domain.
        - 'freq_time': Apply both frequency and time domain processing.
        - 'conv': Apply convolution-like mean pooling to X and optionally to Y.
        - 'threshold': Apply thresholding to X to convert it into a binary format.
        - 'threshold_keep': Apply thresholding to X but retain the original values for further processing.
    - init_strategy : The initialization strategy for dataset.
    - kernel_size : The kernel size for pooling in the 'conv' case. Defaults to (1, 10).
    - threshold : The threshold value for the 'threshold' and 'threshold_keep' strategies. 
                                   Values in X greater than this threshold will be set to 1, others to 0. 
                                   Defaults to 0.65.

    Returns:
    - X : Preprocessed input data, with the transformation applied based on the selected strategy.
    - Y : Preprocessed label data (may be modified in some strategies).
    '''
    
    if strategy == 'freq':
        X = frequency_domain_processing(X=X, sample_rate=extra_info['srate'], time=False)
    elif strategy == 'freq_time':
        X = frequency_domain_processing(X=X, sample_rate=extra_info['srate'], time=True)
    elif strategy == 'conv':
        pooling_layer = AdjustableMeanPooling(kernel_size=kernel_size)
        X = pooling_layer(X)
        if init_strategy != 'simple':
            pooling_layer = AdjustableModePooling(kernel_size=kernel_size)
            Y = pooling_layer(Y)
    elif strategy == 'threshold':
        X = threshold_eeg_data(X=X, threshold=threshold, keep_original=False)
    elif strategy == 'threshold_keep':
        X = threshold_eeg_data(X=X, threshold=threshold, keep_original=True)
    else:
        raise ValueError('Invalid strategy. You can only use [freq, freq_time, conv, threshold, threshold_keep]')

    return X, Y


    


    