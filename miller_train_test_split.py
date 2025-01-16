def train_test_split(X, Y, concat_experiments=False):
    
    n_patients = X.shape[0]

    data  = {}
    
    for i in range(n_patients):
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        for j in range(n_patients):
            if i==j: # Test Sample
                if concat_experiments:
                    test_X.append(X[i][0])
                    test_Y.append(Y[i][0])
                else:
                    test_X.append(X[i][0])
                    test_Y.append(Y[i][0])
                    test_X.append(X[i][1])
                    test_Y.append(Y[i][1])
            else:   # Train Samples
                if concat_experiments:
                    train_X.append(X[j][0])
                    train_Y.append(Y[j][0])
                else:
                    train_X.append(X[j][0])
                    train_Y.append(Y[j][0])
                    train_X.append(X[j][1])
                    train_Y.append(Y[j][1])
        data[i] = {
            'train_X':train_X,
            'train_Y':train_Y,
            'test_X':test_X,
            'test_Y':test_Y
        }
    
    return data