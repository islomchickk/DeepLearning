import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return len(self.X)

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = -1
        if self.shuffle:
            perm = np.random.permutation(len(self.X))
            self.X = self.X[perm]
            self.y = self.y[perm]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        self.batch_id += 1
        if self.batch_id < self.__len__():
            ind = self.batch_id * self.batch_size
            return (self.X[ind: ind + self.batch_size], self.y[ind: ind + self.batch_size])

        raise StopIteration
