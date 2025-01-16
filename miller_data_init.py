import sys
sys.path.append('.')
import numpy as np

from load_miller import *

def normalize_signal(V, scale_uv):
    '''
    Normalize the voltage data using the scale_uv factor for each channel.

    Parameters:
        - V : The raw voltage data with shape (..., C), where C is the number of channels.
        - scale_uv : A vector of scaling factors with length equal to the number of channels.

    Returns:
        - normalized_data : The normalized voltage data.
    '''
    normalized_data = V * scale_uv

    return normalized_data

def dataset_init(strategy = 'simple',
                  binary_classifier = True,
                    concat_experiments = False,
                      dataset_path = 'motor_imagery.npz',
                        sub = 0,
                          channels = np.arange(46), normalize = True):
    '''
    Parameters:

    - strategy: The strategy which is used to create the dataset from the raw data. Valid inputs are:
        
        - simple : The input tensor will have the shape (N, M, Trial, time_points, Channles), 
                    and the output tensor will have the shape (N, M, Trials)
        
        - simple_continues : The input tensor will have the shape (N, M, Trial, time_points, Channles), 
                                and the output tensor will have the shape (N, M, Trial, time_points, Channles) 
        
        - continues : The input tensor will have the shape (N, M, time_points_hat, Channles), 
                        and the output tensor will have the shape (N, M, time_points_hat, Channles)  
        
        - continue_idle : The input tensor will have the shape (N, M, Total_time_points, Channles) 
                            and the output tensor will have the shape (N, M, Total_time_points, Channles)

    - binary_classifier: Defines the values of the label. If it is True, then the labels can be either 0 or 1. 
                         If it is false, then the labels are either 0, 1, 2, or 3. 
                         **Note**: Incase the strategy contains idle states, that is strategy == 'continue_idle', 
                                    then the labels are either 0 (for the idle state), 1, 2, 3, or 4

    - concat_experiments: Defines whether to unfold the tensor on its 2nd dimension or not. 
                            If it is set as True, then the output tensor shapes will be slightly  adjusted and the first and second dimension will be combined into N * M.
                            Otherwise, they will remain the same. 
    
    - dataset_pat: path to the dataset file .npz 

    - sub : constraint on the t_off --> 3000 - sub
                default value is 0
    
    - channels: the array of channels to be used. default value is the first 46 channels.

    - normalize : whether to scale uv the voltages or not. default is set as True.

    Where : 
        - N is the number of patients
        - M is the number of experiments
        - Trials is the number of trails per experiment
        - Channels is the number of electrods
        - time_points is the length of each trail
        - time_points_hat is the length of all trails ( = Trials * time_points)
        - Total_time_points is the total length of the experiment. 
    Returns:
    - X : Input tensor
    - Y : Output tensor
    - extra_info : Additional info used for further pre-processing
    '''

    data = np.load(dataset_path, allow_pickle=True)['dat']

    minimum_n_time_points = 376240


    X = []
    Y = []
    for i in range(7):
        experiment_x = []
        experiment_y = []
        for j in range(2):
            sample, extra_info = None, None
            if strategy == 'simple':
                sample, extra_info = split_classes(data[i][j], sub=sub, channels=channels)
            elif strategy == 'simple_continues':
                sample, extra_info = continues_classes(data[i][j], sub=sub, channels=channels)
            elif strategy == 'continues':
                sample, extra_info = continues_classes(data[i][j], sub=sub, channels=channels)
                sample['V'] = sample['V'].reshape(sample['V'].shape[0] * sample['V'].shape[1], sample['V'].shape[2])
                sample['stim_id'] = sample['stim_id'].reshape(sample['stim_id'].shape[0] * sample['stim_id'].shape[1], sample['stim_id'].shape[2])
            elif strategy == 'continue_idle':
                sample, extra_info = continues_classes_idle(data[i][j], sub=sub, channels=channels)
                n_time_points = sample['V'].shape[0]
                difference_time_points = n_time_points -  minimum_n_time_points
                adjustment_window = difference_time_points//2
                V = sample['V'][adjustment_window:n_time_points - adjustment_window]
                stim_id = sample['stim_id'][adjustment_window:n_time_points - adjustment_window]
                sample['V'] = V
                sample['stim_id'] = stim_id
            else:
                raise   ValueError('Valid inputs for strategy are : simple, simple_continues, continues, or continue_idle')
            
            if normalize:
                sample['V'] = normalize_signal(V=sample['V'], scale_uv=extra_info['scale_uv'][channels])

            experiment_x.append(sample['V'])

            if binary_classifier:
                experiment_y.append(sample['stim_id'])
            else:
                if j==1:
                    if 3 in sample['stim_id']:
                        sample['stim_id'] = sample['stim_id'] + 3
                        sample['stim_id'][sample['stim_id'] == 3] = 0
                    else:
                        sample['stim_id'] = sample['stim_id'] + 2
                experiment_y.append(sample['stim_id'])

        X.append(np.array(experiment_x))
        Y.append(np.array(experiment_y))
    
    
    X = np.array(X)
    Y = np.array(Y)

    if concat_experiments:
        X = np.concatenate([X[:,0], X[:,1]], axis=-2)
        if strategy=='simple':
            Y = np.concatenate([Y[:,0], Y[:,1]], axis=-1)
        else:
            Y = np.concatenate([Y[:,0], Y[:,1]], axis=-2)

    return X, Y, extra_info




        
