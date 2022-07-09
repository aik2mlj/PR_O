# pyright: reportOptionalSubscript=false

from torchaudio.transforms import MelScale
from torch.utils.data import Dataset, DataLoader
from utils import (
    nmat_to_pianotree_repr,
    nmat_to_pr_mat_repr,
    nmat_to_rhy_array,
    compute_pr_mat_feat,
)
from constants import *
from utils import read_split_dict
from dirs import *
import os
import torch
from utils import chd_to_onehot


class DataSampleNpz:
    """
    A pair of song segment stored in .npz format
    containing piano and orchestration versions
    each contains "notes" and "chord" info

    This class aims to get input samples for a single song
    `__getitem__` is used for retrieving ready-made input segments to the model
    it will be called in DataLoader
    """
    def __init__(self, song_fn, load_chord) -> None:
        self.dpath = os.path.join(QUANTIZED_DATA_DIR, song_fn)
        self.fpath_x = os.path.join(self.dpath, "orchestra.npz")
        self.fpath_y = os.path.join(self.dpath, "piano.npz")
        """
        notes (onset_beat, onset_bin, duration, pitch, velocity)
        chord for each beat (root, semitone_bitmap, bass)
        start_table : i-th row indicates the starting row of the "notes" array
            at i-th beat.
        db_pos: an array of downbeat beat_ids

        x: orchestra
        y: piano

        dict : each downbeat corresponds to a SEG_LGTH-long segment
            nmat: note matrix (same format as input npz files)
            pr_mat: piano roll matrix (the format for texture decoder)
            pianotree: pianotree format (used for calculating loss & teacher-forcing)
        """

        # self.notes = None
        # self.chord = None
        # self.start_table = None
        # self.db_pos = None

        # self._nmat_dict = None
        # self._pianotree_dict = None
        # self._pr_mat_dict = None
        # self._feat_dict = None

        # def load(self, use_chord=False):
        #     """ load data """

        # TODO: multiple piano & orchestra versions
        data_x = np.load(self.fpath_x)
        self.notes_x = data_x["notes"]
        self.chord_x = data_x["chord"] if load_chord else None
        self.start_table_x = data_x["start_table"]

        data_y = np.load(self.fpath_y)
        self.notes_y = data_y["notes"]
        self.chord_y = data_y["chord"] if load_chord else None
        self.start_table_y = data_y["start_table"]

        self.db_pos = data_x["db_pos"]

        self._nmat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pianotree_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pr_mat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._feat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))

        self._nmat_dict_y = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pianotree_dict_y = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pr_mat_dict_y = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._feat_dict_y = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of complete 8-beat segments in a song"""
        # NOTE: this is correct because we can feed every 2 bar
        # starting from every downbeat, so there will be overlaps
        if len(self.chord_x) < self.db_pos[-2] + SEG_LGTH:
            return max(len(self.db_pos) - 2, 0)
        elif len(self.chord_x) < self.db_pos[-1] + SEG_LGTH:
            return max(len(self.db_pos) - 1, 0)
        else:
            return len(self.db_pos)

    def note_mat_seg_at_db_x(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        s_ind = self.start_table_x[db]
        e_ind = self.start_table_x[db + SEG_LGTH]
        seg_mats = self.notes_x[s_ind:e_ind]
        return seg_mats.copy()

    def note_mat_seg_at_db_y(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        s_ind = self.start_table_y[db]
        e_ind = self.start_table_y[db + SEG_LGTH]
        seg_mats = self.notes_y[s_ind:e_ind]
        return seg_mats.copy()

    @staticmethod
    def reset_db_to_zeros(note_mat, db):
        note_mat[:, 0] -= db

    @staticmethod
    def format_reset_seg_mat(seg_mat):
        """
        The input seg_mat is (N, 6)
        The output seg_mat is (N, 3). Columns for onset, pitch, duration.
        Onset ranges between range(0, 32).
        """

        output_mat = np.zeros((len(seg_mat), 3), dtype=np.int64)
        output_mat[:, 0] = seg_mat[:, 0] * N_BIN + seg_mat[:, 1]
        output_mat[:, 1] = seg_mat[:, 3]
        output_mat[:, 2] = seg_mat[:, 2]
        return output_mat

    def store_nmat_seg_x(self, db):
        """
        Get note matrix (SEG_LGTH) of orchestra(x) at db position
        """
        if self._nmat_dict_x[db] is not None:
            return

        nmat = self.note_mat_seg_at_db_x(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mat(nmat)
        self._nmat_dict_x[db] = nmat

    def store_nmat_seg_y(self, db):
        """
        Get note matrix (SEG_LGTH) of piano(y) at db position
        """
        if self._nmat_dict_y[db] is not None:
            return

        nmat = self.note_mat_seg_at_db_y(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mat(nmat)
        self._nmat_dict_y[db] = nmat

    def store_prmat_seg_x(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._pr_mat_dict_x[db] is not None:
            return

        prmat = nmat_to_pr_mat_repr(self._nmat_dict_x[db])
        self._pr_mat_dict_x[db] = prmat

    def store_prmat_seg_y(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._pr_mat_dict_y[db] is not None:
            return

        prmat = nmat_to_pr_mat_repr(self._nmat_dict_y[db])
        self._pr_mat_dict_y[db] = prmat

    def store_features_seg_y(self, db):
        """
        Get symbolic features (SEG_LGTH) according to A2S
        """
        if self._feat_dict_y[db] is not None:
            return

        # FIXME: Are these features correct for the whole nmat ?
        rhy = nmat_to_rhy_array(self._nmat_dict_y[db])
        bass_prob, pno_intensity = compute_pr_mat_feat(self._pr_mat_dict_y[db])
        self._feat_dict_y[db] = np.stack([bass_prob, pno_intensity, rhy], -1)

    def store_pno_tree_seg_y(self, db):
        """
        Get PianoTree representation (SEG_LGTH) from nmat
        """
        if self._pianotree_dict_y[db] is not None:
            return

        self._pianotree_dict_y[db] = nmat_to_pianotree_repr(self._nmat_dict_y[db])

    def _store_seg(self, db):
        self.store_nmat_seg_x(db)
        self.store_nmat_seg_y(db)
        self.store_prmat_seg_x(db)
        self.store_prmat_seg_y(db)
        self.store_features_seg_y(db)
        self.store_pno_tree_seg_y(db)

    def _get_item_by_db(self, db):
        """
        Return segments of
            pianotree_y, chord_x, prmat_x, feature_y
        """

        self._store_seg(db)

        # TODO: also return prmat_y for contrastive loss
        # chord
        seg_chd_x = self.chord_x[db:db + SEG_LGTH]
        # prmat
        seg_prmat_x = self._pr_mat_dict_x[db]
        # pianotree
        seg_pno_tree_y = self._pianotree_dict_y[db]
        # symbolic feature
        seg_feat_y = self._feat_dict_y[db]

        return seg_pno_tree_y, seg_chd_x, seg_prmat_x, seg_feat_y

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)

    def get_whole_song_data(self):
        """
        used when inference
        """
        pno_tree_y = []
        chd_x = []
        prmat_x = []
        feat_y = []

        for i in range(0, len(self), 2):
            seg_pno_tree_y, seg_chd_x, seg_prmat_x, seg_feat_y = self[i]
            pno_tree_y.append(seg_pno_tree_y)
            chd_x.append(chd_to_onehot(seg_chd_x))
            prmat_x.append(seg_prmat_x)
            feat_y.append(seg_feat_y)

        pno_tree_y = torch.from_numpy(np.array(pno_tree_y, dtype=np.float32))
        chd_x = torch.from_numpy(np.array(chd_x, dtype=np.float32))
        prmat_x = torch.from_numpy(np.array(prmat_x, dtype=np.float32))
        feat_y = torch.from_numpy(np.array(feat_y, dtype=np.float32))

        return pno_tree_y, chd_x, prmat_x, feat_y


class PianoOrchDataset(Dataset):
    def __init__(self, data_samples):
        super(PianoOrchDataset, self).__init__()

        # a list of DataSampleNpz
        self.data_samples = data_samples

        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)
        self.lgth_cumsum = np.cumsum(self.lgths)

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, index):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > index)[0][0]
        song_item = index - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        song_data = self.data_samples[song_no]
        return song_data[song_item]

    @classmethod
    def load_with_song_paths(cls, song_paths, **kwargs):
        # FIXME: kwargs not used
        data_samples = [
            DataSampleNpz(song_path, load_chord=True) for song_path in song_paths
        ]
        return cls(data_samples)

    @classmethod
    def load_train_and_valid_sets(cls, **kwargs):
        split = read_split_dict(None, None)
        return cls.load_with_song_paths(split[0], **kwargs), cls.load_with_song_paths(
            split[1], **kwargs
        )

    @classmethod
    def load_with_train_valid_paths(cls, tv_song_paths, **kwargs):
        return cls.load_with_song_paths(tv_song_paths[0],
                                        **kwargs), cls.load_with_song_paths(
                                            tv_song_paths[1], **kwargs
                                        )
