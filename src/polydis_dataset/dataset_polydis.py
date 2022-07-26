import sys
import os

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchaudio.transforms import MelScale
from torch.utils.data import Dataset, DataLoader
from .data_sample_polydis import TrainDataSample
from utils import pianotree_pitch_shift, \
    chd_pitch_shift, chd_to_onehot, pr_mat_pitch_shift
from amc_dl.torch_plus import DataLoaders
from audio_sp_modules.my_collate import default_collate
from audio_sp_modules.pitch_shift import pitch_shift_to_spec
from constants import *
from utils import read_split_dict


class AudioMidiDataset(Dataset):
    def __init__(self, data_samples):

        # a list of data_sample.TrainDataSample
        self.data_samples = data_samples

        # a list of complete sample counts in each song
        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)

        self.lgth_cumsum = np.cumsum(self.lgths)

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, item):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > item)[0][0]
        song_item = item - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        # fetch segment
        song_data = self.data_samples[song_no]
        return song_data[song_item]

    @classmethod
    def load_with_song_ids(cls, song_ids, **kwargs):
        """song_ids: a list of int/str"""
        data_samples = [TrainDataSample(song_id, **kwargs) for song_id in song_ids]
        return cls(data_samples=data_samples)

    @classmethod
    def load_with_train_valid_ids(cls, tv_song_ids, **kwargs):
        """tv_song_ids: a tuple of (train_song_ids, valid_song_ids)"""
        return cls.load_with_song_ids(tv_song_ids[0], **kwargs), \
            cls.load_with_song_ids(tv_song_ids[1], **kwargs)

    @classmethod
    def load_train_and_valid_sets(cls, meter, n_subdiv, **kwargs):
        """
        meter: 2, 3, or None.
            To select subset of pieces in duple(2) or triple(3) meter.
            None means select all songs.
        n_subdiv: 2, 3, or None.
            To select subset of pieces having 2 or 3-based subdivision.
            None means all.
        """
        return cls.load_with_train_valid_ids(
            read_split_dict(meter, n_subdiv, is_polydis=True), **kwargs
        )


def collate_fn(batch, device, augment_p):
    """
    The customized collate_fn function used in DataLoader.
    augment_p: the probability of pitch shift. If None, no pitch shift applied.

    The function
    1) pitch-shift augment the symbolic data
    2) batching and converting data device and to cuda possibly.
    3) pitch-shift augment the audio data.
    """
    def sample_with_p():
        return np.random.choice(np.arange(-6, 6), 1, p=augment_p)[0]

    def sample_op(b):
        # b: (pianotree, chord, piano-roll, symbolic feature)
        pt, c, pm, feat = b
        pt = pianotree_pitch_shift(pt, aug_shift)
        c = chd_to_onehot(chd_pitch_shift(c, aug_shift))
        pm = None if pm is None else pr_mat_pitch_shift(pm, aug_shift)
        feat = None if feat is None else feat
        return pt, c, pm, feat

    # random shift degree
    aug_shift = sample_with_p() if augment_p is not None else 0

    # before collate: pitch shifting the symbolic data
    batch = [sample_op(b) for b in batch]

    # collate: default pytorch collate function
    pno_tree, chd, pr_mat, feat = default_collate(batch)

    # after collate: to datatype, device
    pno_tree = pno_tree.long().to(device)
    chd = chd.float().to(device)
    pr_mat = None if pr_mat is None else pr_mat.float().to(device)
    feat = None if feat is None else feat.float().to(device)

    # modified for PR_O model testing
    return pno_tree, chd, pr_mat, feat, pno_tree


def create_data_loader_polydis(
    dataset, batch_size, shuffle, aug_p, device, num_workers=0
):
    """ create a pytorch data loader with customized collate_fn. """
    return DataLoader(
        dataset,
        batch_size,
        shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, device, aug_p)
    )


class AudioMidiDataLoaders(DataLoaders):
    """ Dataloaders containing train and valid dataloaders. """
    def batch_to_inputs(self, batch):
        return batch

    @classmethod
    def get_loaders(
        cls,
        bs_train,
        bs_val,
        train_dataset,
        val_dataset,
        shuffle_train=True,
        shuffle_val=False,
        aug_p=None,
        num_workers=0
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = \
            create_data_loader_polydis(train_dataset, bs_train, shuffle_train,
                               aug_p, device, num_workers)
        val_loader = \
            create_data_loader_polydis(val_dataset, bs_val, shuffle_val, None,
                               device, num_workers)
        return cls(train_loader, val_loader, bs_train, bs_val, device)

    @property
    def train_set(self):
        return self.train_loader.dataset

    @property
    def val_set(self):
        return self.val_loader.dataset


def create_data_loaders_polydis(
    batch_size,
    num_workers=0,
    meter=2,
    n_subdiv=2,
    shuffle_train=True,
    shuffle_val=False,
    **kwargs
):
    """Fast data loaders initialization."""

    train_dataset, valid_dataset = \
        AudioMidiDataset.load_train_and_valid_sets(
            meter, n_subdiv, **kwargs)

    aug_p = AUG_P / AUG_P.sum()

    return AudioMidiDataLoaders.get_loaders(
        batch_size, batch_size, train_dataset, valid_dataset, shuffle_train,
        shuffle_val, aug_p, num_workers
    )
