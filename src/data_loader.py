import torch
from torch.utils.data import DataLoader
from dataset import PianoOrchDataset
from utils import (
    pianotree_pitch_shift,
    chd_pitch_shift,
    chd_to_onehot,
    pr_mat_pitch_shift,
)
from amc_dl.torch_plus import DataLoaders
from audio_sp_modules.my_collate import default_collate
from constants import *


def collate_fn(batch, device, augment_p, type=None):
    """
    The customized collate_fn function used in DataLoader.
    augment_p: the probability of pitch shift. If None, no pitch shift applied.

    The function
    1) pitch-shift augment the symbolic data
    2) batching and converting data device and to cuda possibly.
    """
    def sample_with_p():
        return np.random.choice(np.arange(-6, 6), 1, p=augment_p)[0]

    def sample_op(b):
        # b: (pianotree, chord, piano-roll, symbolic feature, piano-roll_y)
        if type == "pianotree":
            pt_x, pt_y = b
            pt_x = pianotree_pitch_shift(pt_x, aug_shift)
            pt_y = pianotree_pitch_shift(pt_y, aug_shift)
            return pt_x, pt_y
        else:
            pt, c, pm, feat, pm_y = b
            pt = pianotree_pitch_shift(pt, aug_shift)
            c = chd_to_onehot(chd_pitch_shift(c, aug_shift))
            pm = None if pm is None else pr_mat_pitch_shift(pm, aug_shift)
            feat = None if feat is None else feat
            pm_y = None if pm_y is None else pr_mat_pitch_shift(pm_y, aug_shift)
            return pt, c, pm, feat, pm_y

    # random shift degree
    aug_shift = sample_with_p() if augment_p is not None else 0

    # before collate: pitch shifting the symbolic data
    batch = [sample_op(b) for b in batch]

    # collate: default pytorch collate function
    if type == "pianotree":
        pt_x, pt_y = default_collate(batch)
        pt_x = pt_x.long().to(device)
        pt_y = pt_y.long().to(device)
        return pt_x, pt_y
    else:
        pno_tree, chd, pr_mat, feat, prmat_y = default_collate(batch)
        # after collate size: (B: batch size)
        # (B, 32, 16, 6) (B, 8, 36) (B, 32, 128) (B, 32, 3)

        # after collate: to datatype, device, and pitch shift the audio.
        pno_tree = pno_tree.long().to(device)
        chd = chd.float().to(device)
        pr_mat = None if pr_mat is None else pr_mat.float().to(device)
        feat = None if feat is None else feat.float().to(device)
        prmat_y = None if prmat_y is None else prmat_y.float().to(device)

        return pno_tree, chd, pr_mat, feat, prmat_y


class PianoOrchDataLoader(DataLoaders):
    """Dataloaders containing train and valid dataloaders."""
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
        num_workers=0,
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(
            train_dataset,
            bs_train,
            shuffle_train,
            num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, device, aug_p, **kwargs),
        )
        val_loader = DataLoader(
            val_dataset,
            bs_val,
            shuffle_val,
            num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, device, None, **kwargs),
        )
        return cls(train_loader, val_loader, bs_train, bs_val, device)

    @property
    def train_set(self):
        return self.train_loader.dataset

    @property
    def val_set(self):
        return self.val_loader.dataset


def create_data_loaders(
    batch_size,
    num_workers=0,
    meter=2,
    n_subdiv=2,
    shuffle_train=True,
    shuffle_val=False,
    **kwargs
):
    """Fast data loaders initialization."""

    train_dataset, valid_dataset = PianoOrchDataset.load_train_and_valid_sets(**kwargs)

    aug_p = AUG_P / AUG_P.sum()

    return PianoOrchDataLoader.get_loaders(
        batch_size, batch_size, train_dataset, valid_dataset, shuffle_train,
        shuffle_val, aug_p, num_workers, **kwargs
    )
