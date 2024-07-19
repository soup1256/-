import os
import sys
import threading
import queue
import random

import imageio
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data._utils import collate, signal_handling, ExceptionWrapper
from torch.utils.data._utils.worker import _worker_loop
from torch.utils.data._utils.pin_memory import _pin_memory_loop
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter

from TestCode.code import data

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    signal_handling._set_worker_signal_handlers()
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    if init_fn is not None:
        init_fn(worker_id)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and getattr(dataset, 'train', False):
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)
            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

class _MSDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self.dataset = loader.dataset
        self.noise_g = loader.noise_g
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queues[i],
                        self.worker_result_queue,
                        self.collate_fn,
                        self.noise_g,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        try:
            indices = next(self.sample_iter)
            self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
            self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
            self.batches_outstanding += 1
            self.send_idx += 1
        except StopIteration:
            for q in self.index_queues:
                q.put(None)

class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.noise_g = args.noise_g

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MSDataLoaderIter(self)
        
class DocumentDenoisingDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.noisy_dir = os.path.join(root, 'noisy')
        self.clean_dir = os.path.join(root, 'clean')
        self.noisy_images = sorted(os.listdir(self.noisy_dir))
        self.clean_images = sorted(os.listdir(self.clean_dir))
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        noisy_image = imageio.imread(noisy_path)
        clean_image = imageio.imread(clean_path)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        noisy_image = torch.tensor(noisy_image).float().unsqueeze(0)
        clean_image = torch.tensor(clean_image).float().unsqueeze(0)

        return noisy_image, clean_image, self.noisy_images[idx]
