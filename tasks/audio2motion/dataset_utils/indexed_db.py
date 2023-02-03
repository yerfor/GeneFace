import numpy as np
import pickle
from copy import deepcopy


class IndexedDataset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        try:
            item = pickle.loads(b)
        except:
            return None
        if isinstance(item, bytes):
            return None
        if isinstance(item, int):
            return None
        return item

    def __len__(self):
        return len(self.data_offsets) - 1