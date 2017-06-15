import numpy as np

class Data:
    """
    Convenience class for loading in data from npz file
    To load data run Data(filename).get_data()
	To save data run
	Data(filename, *data).write_csvs()
    """
    def __init__(self, name, X_train=None, X_test=None, y_train=None, y_test=None, labels=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = labels
        self.name = name
        self.seed = 23144235
        self.train_split = 0.8 #80% training data for shuffle
        np.random.seed(self.seed)

    def write_csvs(self):
        np.savez(self.name, *[self.X_train, self.X_test, self.y_train, self.y_test, self.labels])

    def get_data(self):
        f = np.load(self.name + '.npz')
        self.X_train, self.X_test, self.y_train, self.y_test, self.labels = \
            f['arr_0'], f['arr_1'], f['arr_2'], f['arr_3'], f['arr_4']
        return self.X_train, self.X_test, self.y_train, self.y_test, self.labels

    def get_shuffled_data(self):
        X_train_test = np.vstack([self.X_train, self.X_test])
        y_train_test = np.vstack([self.y_train, self.y_test])
        indices = np.array(range(X_train_test.shape[0]))
        np.random.shuffle(indices)
        X_train_test = X_train_test[indices]
        y_train_test = y_train_test[indices]
        N_train = int(self.train_split * X_train_test.shape[0])
        X_train = X_train_test[:N_train]
        X_test = X_train_test[N_train:]
        y_train = y_train_test[:N_train]
        y_test = y_train_test[N_train:]
        return X_train, X_test, y_train, y_test, self.labels, self.seed


