import numpy as np

fname = 'motor_imagery.npz'

def split_classes(data, sub = 0, channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0
    - channels: the array of channels to be used. default value is the first 46 channels.

    Returns:
    - discrete_data: Data dictionary with discrete labels.
    - extra_info: extra information to be used for further pre-processing
    """

    # Initialize dictionaries
    discrete_data = {'V': [], 'stim_id': [], 't_on': [], 't_off': []}
    # Extra Information for preprocessing 
    extra_info = {'srate': data['srate'],
                                   'scale_uv':data['scale_uv'],
                                     'locs': data['locs'],
                                       'hemisphere':data['hemisphere'],
                                         'lobe':data['lobe'],
                                           'gyrus':data['gyrus'],
                                             'Brodmann_Area':data['Brodmann_Area']}

    # Iterate through the stimulus IDs and split the data
    for i, stim_id in enumerate(data['stim_id']):
        t_on = data['t_on'][i]
        t_off = data['t_off'][i] - sub
        discrete_data['V'].append(data['V'][t_on:t_off][:,channels])
        discrete_data['t_on'].append(t_on)
        discrete_data['t_off'].append(t_off)
        discrete_data['stim_id'].append(stim_id)

    # Convert lists to numpy arrays
    discrete_data['V'] = np.array(discrete_data['V'])
    discrete_data['t_on'] = np.array(discrete_data['t_on'])
    discrete_data['t_off'] = np.array(discrete_data['t_off'])
    discrete_data['stim_id'] = np.array(discrete_data['stim_id'])
    discrete_data['stim_id'] = (discrete_data['stim_id']==12).astype(int)

    return discrete_data, extra_info

def continues_classes(data, sub = 0, channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0
    - channels: the array of channels to be used. default value is the first 46 channels.

    Returns:
    - continues_data: Data dictionary for class 1.
    - extra_info: extra information to be used for further pre-processing
    """

    # Initialize dictionaries for the two classes
    continues_data = {'V': [],'stim_id': [], 't_on': [], 't_off': []}
    # Extra Information for preprocessing 
    extra_info = {'srate': data['srate'],
                                   'scale_uv':data['scale_uv'],
                                     'locs': data['locs'],
                                       'hemisphere':data['hemisphere'],
                                         'lobe':data['lobe'],
                                           'gyrus':data['gyrus'],
                                             'Brodmann_Area':data['Brodmann_Area']}

    # Iterate through the stimulus IDs and split the data
    for i, stim_id in enumerate(data['stim_id']):
        t_on = data['t_on'][i]
        t_off = data['t_off'][i] - sub
        continues_data['V'].append(data['V'][t_on:t_off][:,channels])
        continues_data['t_on'].append(t_on)
        continues_data['t_off'].append(t_off)
        continues_data['stim_id'].append(np.zeros(data['V'][t_on:t_off][:,channels].shape))

    # Convert lists to numpy arrays
    continues_data['V'] = np.array(continues_data['V'])
    continues_data['t_on'] = np.array(continues_data['t_on'])
    continues_data['t_off'] = np.array(continues_data['t_off'])
    continues_data['stim_id'] = np.array(continues_data['stim_id'])

    return continues_data, extra_info


def continues_classes_idle(data, sub = 0, channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0
    - channels: the array of channels to be used. default value is the first 46 channels.

    Returns:
    - continues_data: Data dictionary for class 1.
    - extra_info: extra information to be used for further pre-processing
    """

    # Initialize dictionaries for the two classes
    voltages = data['V'][:,channels]
    continues_data = {'V': voltages, 't_on': data['t_on'], 't_off': data['t_off']}

    # Extra Information for preprocessing 
    extra_info = {'srate': data['srate'],
                                   'scale_uv':data['scale_uv'],
                                     'locs': data['locs'],
                                       'hemisphere':data['hemisphere'],
                                         'lobe':data['lobe'],
                                           'gyrus':data['gyrus'],
                                             'Brodmann_Area':data['Brodmann_Area']}

    stim_id = np.zeros(voltages.shape)

    for i in range(data['t_on'].shape[0]):
        t_on = data['t_on'][i]
        t_off = data['t_off'][i] - sub
        if  data['stim_id'][i]==11:
            label = 1
        else:
            label = 2
        stim_id[t_on:t_off, :] = label

    continues_data['stim_id'] = stim_id
    
    return continues_data, extra_info