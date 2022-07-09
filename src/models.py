from amc_dl.torch_plus import PytorchModel
from amc_dl.torch_plus.train_utils import get_zs_from_dists, kl_with_normal
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
from dl_modules import (
    ChordEncoder, ChordDecoder, PianoTreeDecoder, TextureEncoder, FeatDecoder, NaiveNN
)


class PianoReductionVAE(PytorchModel):
    """
    The proposed piano reduction VAE model.
    """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_chd', 'kl_sym', 'chord_l',
        'root_l', 'chroma_l', 'bass_l', 'feat_l', 'bass_feat_l', 'int_feat_l',
        'rhy_feat_l', 'beta'
    ]

    def __init__(
        self, name, device, chord_enc: ChordEncoder, chord_dec: ChordDecoder,
        prmat_enc: TextureEncoder, diffusion_nn: NaiveNN, feat_dec: FeatDecoder,
        pianotree_dec: PianoTreeDecoder
    ):

        super(PianoReductionVAE, self).__init__(name, device)

        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

        # texture encoder + diffusion model = symbolic encoder
        self.prmat_enc = prmat_enc
        self.diffusion_nn = diffusion_nn  # TODO

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = feat_dec
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = pianotree_dec

    @property
    def z_chd_dim(self):
        return self.chord_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    def run(self, pno_tree_y, chd, pr_mat, feat_y, tfr1, tfr2, tfr3):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # chord representation
        dist_chd = self.chord_enc(chd)
        z_chd = dist_chd.rsample()

        # symbolic-texture representation
        dist_x_sym = self.prmat_enc(pr_mat)
        # z_x_sym = dist_x_sym.rsample()
        # print(dist_x_sym)

        # diffusion model: transform z_x_txt to z_y_txt
        dist_sym = self.diffusion_nn(dist_x_sym)
        # print(dist_sym)
        z_sym = dist_sym.rsample()  # this is z_y_txt

        # z
        z = torch.cat([z_chd, z_sym], -1)

        # reconstruction of chord progression
        recon_root, recon_chroma, recon_bass = self.chord_dec(z_chd, False, tfr3, chd)

        # reconstruction of symbolic feature from z_y_txt
        recon_feat = self.feat_dec(z_sym, False, tfr1, feat_y)

        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree_y)

        # pianotree decoder
        recon_pitch, recon_dur = self.pianotree_dec(
            z, False, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, feat_emb
        )

        return (
            recon_pitch, recon_dur, recon_root, recon_chroma, recon_bass, recon_feat,
            dist_chd, dist_sym
        )

    def loss_function(
        self, pno_tree_y, feat_y, chd, recon_pitch, recon_dur, recon_root, recon_chroma,
        recon_bass, recon_feat, dist_chd, dist_sym, beta, weights
    ):
        """Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = self.pianotree_dec.recon_loss(
            pno_tree_y, recon_pitch, recon_dur, weights, False
        )

        # chord recon loss
        chord_l, root_l, chroma_l, bass_l = self.chord_dec.recon_loss(
            chd, recon_root, recon_chroma, recon_bass
        )

        # feature prediction loss
        feat_l, bass_feat_l, int_feat_l, rhy_feat_l = self.feat_dec.recon_loss(
            feat_y, recon_feat
        )

        # kl losses
        kl_chd = kl_with_normal(dist_chd)
        kl_sym = kl_with_normal(dist_sym)
        kl_l = beta * (kl_chd + kl_sym)

        # TODO: contrastive loss

        loss = pno_tree_l + chord_l + feat_l + kl_l

        return (
            loss, pno_tree_l, pitch_l, dur_l, kl_l, kl_chd, kl_sym, chord_l, root_l,
            chroma_l, bass_l, feat_l, bass_feat_l, int_feat_l, rhy_feat_l, beta
        )

    def loss(
        self,
        pno_tree_y,
        chd,
        pr_mat,
        feat_y,
        tfr1,
        tfr2,
        tfr3,
        beta=0.1,
        weights=(1, 0.5)
    ):
        """
        Forward path during training with loss computation.

        :param pno_tree: (B, 32, 16, 6) ground truth for teacher forcing
        :param chd: (B, 8, 36) chord input
        :param pr_mat: (B, 32, 128) (with proper corruption) symbolic input.
        :param feat: (B, 32, 3) ground truth for teacher forcing
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: teacher forcing ratio 3 (for chord decoder)
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        (
            recon_pitch, recon_dur, recon_root, recon_chroma, recon_bass, recon_feat,
            dist_chd, dist_sym
        ) = self.run(pno_tree_y, chd, pr_mat, feat_y, tfr1, tfr2, tfr3)

        return self.loss_function(
            pno_tree_y, feat_y, chd, recon_pitch, recon_dur, recon_root, recon_chroma,
            recon_bass, recon_feat, dist_chd, dist_sym, beta, weights
        )

    @classmethod
    def init_model(cls, z_chd_dim=128, z_sym_dim=192, model_path=None):
        """Fast model initialization."""

        name = 'PianoReductionVAE'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        prmat_enc = TextureEncoder(z_dim=z_sym_dim)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(
            name, device, chord_enc, chord_dec, prmat_enc, diffusion_nn, feat_dec,
            pianotree_dec
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)
        return model

    def inference(self, chord, pr_mat):
        """
        Forward path during inference.

        :param chord: (B, 8, 36) chord input
        :param pr_mat: (B, 32, 128) symbolic piano roll matrices.
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z_chd = self.chord_enc(chord).mean

            dist_x_sym = self.prmat_enc(pr_mat)
            # z_x_sym = dist_x_sym.mean  # FIXME: is this correct?
            dist_sym = self.diffusion_nn(dist_x_sym)
            z_sym = dist_sym.mean  # this is z_y_txt

            z = torch.cat([z_chd, z_sym], -1)

            recon_feat = self.feat_dec(z_sym, True, 0., None)
            feat_emb = self.feat_emb_layer(recon_feat)
            recon_pitch, recon_dur = self.pianotree_dec(
                z, True, None, None, 0., 0., feat_emb
            )

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(), recon_dur.cpu())[0]
        return pred
