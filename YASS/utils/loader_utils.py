from torch._six import int_classes as _int_classes
import torch
from torch.utils.data.sampler import Sampler
from multiprocessing.dummy import Pool as ThreadPool
import itertools


class CustomRandomSampler(Sampler):
    '''
    Samples elements randomly, without replacement. 
    This sampling only shuffles within epoch intervals of the dataset 
    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int) : Number of epochs in the train dataset
        num_workers (int) : Number of workers to use for generating iterator
    '''

    def __init__(self, data_source, num_epochs, num_workers, weights=None, replacement=True):
        self.data_source = data_source
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.datalen = len(data_source)
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        iter_array = []
        pool = ThreadPool(self.num_workers)

        def get_randperm(i):
            if self.weights is None:
                return torch.randperm(self.datalen).tolist()
            # self.weights = torch.tensor(self.weights, dtype=torch.double)
            return torch.multinomial(torch.tensor(self.weights, dtype=torch.double), self.datalen, self.replacement).tolist()
        iter_array = list(itertools.chain.from_iterable(
            pool.map(get_randperm, range(self.num_epochs))))
        pool.close()
        pool.join()
        return iter(iter_array)

    def __len__(self):
        return len(self.data_source)


class CustomBatchSampler(object):
    '''
    Wraps another custom sampler with epoch intervals 
    to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``
        epoch_size : Number of items in an epoch
    '''

    def __init__(self, sampler, batch_size, drop_last, epoch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError('sampler should be an instance of '
                             'torch.utils.data.Sampler, but got sampler={}'
                             .format(sampler))
        if (not isinstance(batch_size, _int_classes) 
                    or isinstance(batch_size, bool) 
                    or batch_size <= 0):
            raise ValueError('batch_size should be a positive integral value, '
                             'but got batch_size={}'.format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError('drop_last should be a boolean value, but got '
                             'drop_last={}'.format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch_size = epoch_size
        self.num_epochs = len(self.sampler)/self.epoch_size

        if self.drop_last:
            self.num_batches_per_epoch = self.epoch_size // self.batch_size
        else:
            self.num_batches_per_epoch = (
                self.epoch_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        epoch_ctr = 0
        for idx in self.sampler:
            epoch_ctr += 1
            batch.append(int(idx))
            if len(batch) == self.batch_size or epoch_ctr == self.epoch_size:
                yield batch
                batch = []
                if epoch_ctr == self.epoch_size:
                    epoch_ctr = 0

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.num_epochs * self.num_batches_per_epoch
