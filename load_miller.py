import numpy as np

fname = 'motor_imagery.npz'
# # dat1 actual 
# # dat2 imagery
# data = np.load(fname, allow_pickle=True)['dat']

def split_classes(data, stim_id_1=11, stim_id_2=12, sub = 0, binary_stim_id = True , channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - stim_id_1: Stimulus ID for class 1. 11 = tongue, 12 = hand 
    - stim_id_2: Stimulus ID for class 2.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0

    Returns:
    - continues_data: Data dictionary for class 1.
    - hand_data: Data dictionary for class 2.
    - extra_info: extra information to be used for further pre-processing
    """

    # Initialize dictionaries for the two classes
    continues_data = {'V': [], 'stim_id': [], 't_on': [], 't_off': []}
    hand_data = {'V': [], 'stim_id': [], 't_on': [], 't_off': []}
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
        if stim_id == stim_id_1:
            continues_data['V'].append(data['V'][t_on:t_off][:,channels])
            continues_data['t_on'].append(t_on)
            continues_data['t_off'].append(t_off)
            continues_data['stim_id'].append(stim_id)
        elif stim_id == stim_id_2:
            hand_data['V'].append(data['V'][t_on:t_off][:,channels])
            hand_data['t_on'].append(t_on)
            hand_data['t_off'].append(t_off)
            hand_data['stim_id'].append(stim_id)

    # Convert lists to numpy arrays
    continues_data['V'] = np.array(continues_data['V'])
    continues_data['t_on'] = np.array(continues_data['t_on'])
    continues_data['t_off'] = np.array(continues_data['t_off'])
    continues_data['stim_id'] = np.array(continues_data['stim_id'])

    hand_data['V'] = np.array(hand_data['V'])
    hand_data['t_on'] = np.array(hand_data['t_on'])
    hand_data['t_off'] = np.array(hand_data['t_off'])
    hand_data['stim_id'] = np.array(hand_data['stim_id'])


    if binary_stim_id:
        continues_data['stim_id'] = 1 - (continues_data['stim_id']==11).astype(int)
        hand_data['stim_id'] = (hand_data['stim_id']==12).astype(int)

    return continues_data, hand_data, extra_info

def continues_classes(data, stim_id_1=11, stim_id_2=12, sub = 0, channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - stim_id_1: Stimulus ID for class 1. 11 = tongue, 12 = hand 
    - stim_id_2: Stimulus ID for class 2.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0

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
        if stim_id == stim_id_1:
            continues_data['V'].append(data['V'][t_on:t_off][:,channels])
            continues_data['t_on'].append(t_on)
            continues_data['t_off'].append(t_off)
            continues_data['stim_id'].append(np.zeros(data['V'][t_on:t_off][:,channels].shape))
        elif stim_id == stim_id_2:
            continues_data['V'].append(data['V'][t_on:t_off][:,channels])
            continues_data['t_on'].append(t_on)
            continues_data['t_off'].append(t_off)
            continues_data['stim_id'].append(np.ones(data['V'][t_on:t_off][:,channels].shape))

    # Convert lists to numpy arrays
    continues_data['V'] = np.array(continues_data['V'])
    continues_data['t_on'] = np.array(continues_data['t_on'])
    continues_data['t_off'] = np.array(continues_data['t_off'])
    continues_data['stim_id'] = np.array(continues_data['stim_id'])

    return continues_data, extra_info


def continues_classes_idle(data, stim_id_1=11, stim_id_2=12, sub = 0, stim_id_idle = 13, channels = np.arange(46)):
    """
    Parameters:
    - data: Dictionary containing the data.
    - stim_id_1: Stimulus ID for class 1. 11 = tongue, 12 = hand 
    - stim_id_2: Stimulus ID for class 2.
    - stim_id_2: Stimulus ID for IDLE state.
    - sub : constraint on the t_off --> 3000 - sub
        default value is 0

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
        t_off = data['t_off'][i]
        if  data['stim_id'][i]==11:
            label = 1
        else:
            label = 2
        stim_id[t_on:t_off, :] = label

    continues_data['stim_id'] = stim_id
    
    return continues_data, extra_info