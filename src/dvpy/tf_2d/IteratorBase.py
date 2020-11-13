# System
import threading

# Third Party
import numpy as np

#


class IteratorBase(object):

    __slots__ = [
        "N",          
        "batch_size",
        "slice_num",
        "shuffle",
        "batch_index",
        "total_batches_seen",
        "lock",
        "index_generator",
    ]

    def __init__(self, N, batch_size, slice_num, shuffle, seed):
        self.N = N  # the number of total cases
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.shuffle = shuffle # whether the index_list is randomized
        self.seed = seed

        self.batch_index = 0 
        self.total_batches_seen = 0 # count how many batches has been input
        self.lock = threading.Lock()


        self.index_generator = self._flow_index()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # ensure self.batch_index is 0

        self.reset()
        while True:
            if self.batch_index == 0:
                patient_list = np.random.permutation(self.N)
                index_array = []
                for p in patient_list:
                    if self.shuffle == True:
                        slice_list = np.random.permutation(self.slice_num)
                    else:
                        slice_list = np.arange(self.slice_num)
                    for s in slice_list:
                        index_array.append([p,s])

                index_array = np.asarray(index_array)
                print(index_array)


            total_slice = self.N * self.slice_num
            current_index = (self.batch_index * self.batch_size) % total_slice
            if total_slice >= current_index + self.batch_size:   # the total number of cases is adequate for next loop
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = total_slice - current_index  # not adequate, should reduce the batch size
                self.batch_index = 0
            self.total_batches_seen += 1
           
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
