import sys
sys.path.append('.')
import numpy as np

from load_miller import *

def dataset_init(strategy = 'simple',
                  binary_classifier = True,
                    concat_experiments = False,
                      dataset_path = 'motor_imagery.npz',
                        sub = 0,
                          channels = np.arange(46) ):
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
            elif strategy == 'continue_idle':
                sample, extra_info = continues_classes_idle(data[i][j], sub=sub, channels=channels)
            else:
                raise   ValueError('Valid inputs for strategy are : simple, simple_continues, continues, or continue_idle')
                
            sample, extra_info = split_classes(data[i][j], sub=sub, channels=channels)
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
        X = X.reshape(-1, *X.shape[2:])
        Y = Y.reshape(-1, *Y.shape[2:])

    return X, Y, extra_info




        
