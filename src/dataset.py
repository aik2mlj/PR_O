# pyright: reportOptionalSubscript=false

from torchaudio.transforms import MelScale
from torch.utils.data import Dataset, DataLoader
from .utils import (
    nmat_to_pianotree_repr,
    nmat_to_pr_mat_repr,
    nmat_to_rhy_array,
    compute_pr_mat_feat,
)
from .constants import *
from .utils import read_split_dict


class DataSampleNpz:
    """
    A single song segment stored in .npz format
    contains "notes" and "chord"

    This class aims to get input samples for a single song
    `_get_item_by_db` is used for retrieving ready-made inputs to the model
    """

    def __init__(self, fpath, use_chord) -> None:
        self.fpath = fpath
        """
        notes (onset_beat, onset_bin, duration, pitch, velocity)
        chord for each beat (root, semitone_bitmap, bass)
        start_table : i-th row indicates the starting row of the "notes" array at i-th beat.
        db_pos: an array of downbeat beat_ids

        dict : each downbeat corresponds to a SEG_LGTH-long batch
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

        data = np.load(self.fpath)
        self.notes = data["notes"]
        self.chord = data["chord"] if use_chord else None
        self.start_table = data["start_table"]
        self.db_pos = data["db_pos"]

        self._nmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pianotree_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pr_mat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._feat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of 8-beat segments in a song"""
        # NOTE: this is correct because we can feed every 2 bar starting from every downbeat
        # so there will be overlaps
        return len(self.db_pos)

    def note_mat_batch_at_db(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        s_ind = self.start_table[db]
        e_ind = self.start_table[db + SEG_LGTH]
        seg_mats = self.notes[s_ind, e_ind]
        return seg_mats

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

    def store_nmat_seg(self, db):
        """
        Get note matrix (SEG_LGTH) at db position
        """
        if self._nmat_dict[db] is not None:
            return

        nmat = self.note_mat_batch_at_db(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mat(nmat)
        self._nmat_dict[db] = nmat

    def store_prmat_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._pr_mat_dict[db] is not None:
            return

        prmat = nmat_to_pr_mat_repr(self._nmat_dict[db])
        self._pr_mat_dict[db] = prmat

    def store_features_seg(self, db):
        """
        Get symbolic features (SEG_LGTH) according to A2S
        """
        if self._feat_dict[db] is not None:
            return

        # FIXME: Are these features correct for the whole nmat ?
        rhy = nmat_to_rhy_array(self._nmat_dict[db])
        bass_prob, pno_intensity = compute_pr_mat_feat(self._pr_mat_dict[db])
        self._feat_dict[db] = np.stack([bass_prob, pno_intensity, rhy], -1)

    def store_pno_tree_seg(self, db):
        """
        Get PianoTree representation (SEG_LGTH) from nmat
        """
        if self._pianotree_dict[db] is not None:
            return

        self._pianotree_dict[db] = nmat_to_pianotree_repr(self._nmat_dict[db])

    def _store_seg(self, db):
        self.store_nmat_seg(db)
        self.store_prmat_seg(db)
        self.store_features_seg(db)
        self.store_pno_tree_seg(db)

    def _get_item_by_db(self, db):
        """
        Return segments of
            pianotree, chord, prmat, feature
        """
        self._store_seg(db)

        # chord
        seg_chd = self.chord[db : db + SEG_LGTH]
        # pianotree
        seg_pno_tree = self._pianotree_dict[db]
        # prmat
        seg_prmat = self._pr_mat_dict[db]
        # symbolic feature
        seg_feat = self._feat_dict[db]

        return seg_pno_tree, seg_chd, seg_prmat, seg_feat


class PianoOrchDataset(Dataset):
    def __init__(self, data_samples, use_chord=False):
        super(PianoOrchDataset, self).__init__()

        # a list of DataSampleNpz
        self.data_samples = data_samples
        self.use_chord = use_chord

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
    def load_with_song_paths(cls, song_paths, use_chord=False):
        data_samples = [DataSampleNpz(song_path, use_chord) for song_path in song_paths]
        return cls(data_samples, use_chord)

    @classmethod
    def load_train_and_valid_sets(cls, use_chord=False):
        split = read_split_dict(None, None)
        return cls.load_with_song_paths(split[0], use_chord), cls.load_with_song_paths(
            split[1], use_chord
        )
