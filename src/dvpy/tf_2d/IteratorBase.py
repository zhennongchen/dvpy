# System
import threading

# Third Party
import numpy as np

#


class IteratorBase(object):

    __slots__ = [
        "N",          
        "batch_size",
        "shuffle",
        "batch_index",
        "total_batches_seen",
        "lock",
        "index_generator",
    ]

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N  # the number of total cases
        self.batch_size = batch_size
        self.shuffle = shuffle # whether the index_list is randomized

        self.batch_index = 0 
        self.total_batches_seen = 0 # count how many batches has been input
        self.lock = threading.Lock()

        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while True:
            print('batch index = ',self.batch_index)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:   
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)
            print('index array = ',index_array)

            current_index = (self.batch_index * batch_size) % N
            # Should this be >, rather than >=?
            # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/preprocessing/image.py#L788
            if N >= current_index + batch_size:   # the total number of cases is adequate for next loop
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index  # not adequate, should reduce the batch size
                self.batch_index = 0
            self.total_batches_seen += 1
            print('current_index = ',current_index)
            print('current batch size = ',current_batch_size)
            yield (
                index_array[current_index : current_index + current_batch_size],
                current_index,
                current_batch_size,
            )

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        # self.next must be defined in the child class.
        return self.next(*args, **kwargs)
