import h5py
import torch
import torch.utils.data as data
import numpy as np

class KITTI(data.Dataset):
    def __init__(self, datafile, sourcefile, nt = 10):
        self.datafile = datafile
        self.sourcefile = sourcefile

        with h5py.File(self.datafile, 'r') as f:
            key = list(f.keys())[0]
            self.X = f[key][:] #X will be like (n_images, nb_cols, nb_rows, nb_channels)
        with h5py.File(self.sourcefile, 'r') as f:
            key = list(f.keys())[0]
            self.sources = f[key][:]

        self.nt = nt

        cur_loc = 0
        possible_starts = []
        while cur_loc < self.X.shape[0] - self.nt + 1:
            if self.sources[cur_loc] == self.sources[cur_loc + self.nt - 1]:
                possible_starts.append(cur_loc)
                cur_loc += self.nt
            else:
                cur_loc += 1
        self.possible_starts = possible_starts

    def __getitem__(self, index):
        loc = self.possible_starts[index]
        seq = self.preprocess(self.X[loc:loc+self.nt])
        return seq

    def __len__(self):
        return len(self.possible_starts)

    def preprocess(self, X):
        return X.astype(np.float32) / 255.
