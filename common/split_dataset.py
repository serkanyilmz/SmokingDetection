import random

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]

    random.shuffle(idx0); random.shuffle(idx1)

    split0 = int(len(idx0) * (1 - test_size))
    split1 = int(len(idx1) * (1 - test_size))

    train_idx = idx0[:split0] + idx1[:split1]
    test_idx = idx0[split0:] + idx1[split1:]

    random.shuffle(train_idx); random.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]

    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test

def train_val_test_split(X, y, train_size=0.6, val_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]

    random.shuffle(idx0); random.shuffle(idx1)

    split0_train = int(len(idx0) * train_size)
    split0_val   = int(len(idx0) * (train_size + val_size))

    split1_train = int(len(idx1) * train_size)
    split1_val   = int(len(idx1) * (train_size + val_size))

    train_idx = idx0[:split0_train] + idx1[:split1_train]
    val_idx   = idx0[split0_train:split0_val] + idx1[split1_train:split1_val]
    test_idx  = idx0[split0_val:] + idx1[split1_val:]

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    X_val   = [X[i] for i in val_idx]
    X_test  = [X[i] for i in test_idx]

    y_train = [y[i] for i in train_idx]
    y_val   = [y[i] for i in val_idx]
    y_test  = [y[i] for i in test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test

