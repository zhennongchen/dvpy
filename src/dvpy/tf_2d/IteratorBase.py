# System
import threading

# Third Party
import numpy as np
import random

#


class IteratorBase(object):

    __slots__ = [
        "N",          
        "slice_num",
        "batch_size",
        "patients_in_one_batch",
        "shuffle",
        "batch_index",
        "total_batches_seen",
        "lock",
        "index_generator",
    ]

    def __init__(self, N, slice_num, batch_size, patients_in_one_batch, shuffle, seed):
        self.N = N  # the number of total cases
        self.slice_num = slice_num # num of slices in each case
        self.batch_size = batch_size # num of slices in each batch
        self.patients_in_one_batch = patients_in_one_batch # num of cases in each batch
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

                if self.shuffle == True: # put several cases into one batch instead of just one case
                    new_index_array = []
                    slices_in_one_group = self.patients_in_one_batch * self.slice_num
                    for i in range(0,int(self.N / self.patients_in_one_batch)):
                        g = index_array[slices_in_one_group * i:slices_in_one_group * (i+1)]
                        random.shuffle(g)
                        new_index_array.extend(g)
                    index_array = new_index_array
  
                index_array = np.asarray(index_array)
                

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
