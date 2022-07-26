from amc_dl.torch_plus import PytorchModel
from amc_dl.torch_plus.train_utils import get_zs_from_dists, kl_with_normal
import torch
from torch import nn
from torch.distributions import Normal
from dl_modules import (
    ChordEncoder, ChordDecoder, PianoTreeDecoder, TextureEncoder, FeatDecoder, NaiveNN
)
from utils import (
    load_pretrained_txt_enc, retrieve_midi_from_chd, estx_to_midi_file, onehot_to_chd,
    retrieve_midi_from_prmat
)


class PianoReductionVAE_contrastive(PytorchModel):
    """
    The proposed piano reduction VAE model.
    """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_chd', 'kl_sym', 'chord_l',
        'root_l', 'chroma_l', 'bass_l', 'feat_l', 'bass_feat_l', 'int_feat_l',
        'rhy_feat_l', 'contra_l', 'beta'
    ]

    def __init__(
        self,
        name,
        device,
        chord_enc: ChordEncoder,
        chord_dec: ChordDecoder,
        prmat_enc: TextureEncoder,
        diffusion_nn: NaiveNN,
        feat_dec: FeatDecoder,
        pianotree_dec: PianoTreeDecoder,
        is_pretrained_prmat_enc,
        # for contrastive learning
        pt_prmat_y_enc: TextureEncoder,
    ):

        super(PianoReductionVAE_contrastive, self).__init__(name, device)

        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

        # texture encoder + diffusion model = symbolic encoder
        self.prmat_enc = prmat_enc
        self.diffusion_nn = diffusion_nn  # TODO

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = feat_dec
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = pianotree_dec

        self.is_pretrained_prmat_enc = is_pretrained_prmat_enc

        # contrastive learning: pretrained prmat_y encoder
        self.pt_prmat_y_enc = pt_prmat_y_enc

    @property
    def z_chd_dim(self):
        return self.chord_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    def run(self, pno_tree_y, chd, pr_mat, feat_y, prmat_y, tfr1, tfr2, tfr3):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # chord representation
        dist_chd = self.chord_enc(chd)
        z_chd = dist_chd.rsample()

        # symbolic-texture representation
        if self.is_pretrained_prmat_enc:
            with torch.no_grad():
                dist_x_sym = self.prmat_enc(pr_mat)
        else:
            dist_x_sym = self.prmat_enc(pr_mat)
        # z_x_sym = dist_x_sym.rsample()
        # print(dist_x_sym)

        # diffusion model: transform z_x_txt to z_y_txt
        dist_sym = self.diffusion_nn(dist_x_sym)
        z_sym = dist_sym.rsample()  # this is z_y_txt

        # contrastive learning TODO
        with torch.no_grad():
            dist_y_sym = self.pt_prmat_y_enc(prmat_y)
        z_y_sym = dist_y_sym.rsample()  # FIXME: correct?

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
            dist_chd, dist_sym, z_sym, z_y_sym
        )

    def loss_function(
        self, pno_tree_y, feat_y, chd, recon_pitch, recon_dur, recon_root, recon_chroma,
        recon_bass, recon_feat, dist_chd, dist_sym, z_sym, z_y_sym, beta, weights
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

        # contrastive loss
        contra_l = self.contra_loss(z_sym, z_y_sym)

        loss = pno_tree_l + chord_l + feat_l + kl_l + contra_l

        return (
            loss, pno_tree_l, pitch_l, dur_l, kl_l, kl_chd, kl_sym, chord_l, root_l,
            chroma_l, bass_l, feat_l, bass_feat_l, int_feat_l, rhy_feat_l, contra_l,
            beta
        )

    @staticmethod
    def contra_loss(z1: torch.Tensor, z2: torch.Tensor, T=0.07):
        """
        compute the contrastive loss within one batch
        for each z1, use corresponding z2 as positive, other z2 as negative
        :param z1, z2: (#, 256)
        :param T: temperature
        """
        assert z1.shape == z2.shape
        bs = z1.shape[0]
        dim = z1.shape[1]
        z1_expand = z1.expand(bs, bs, dim).transpose(0, 1)
        z2_expand = z2.expand(bs, bs, dim)
        cos = torch.nn.CosineSimilarity(dim=2)
        cossim_mat = cos(z1_expand, z2_expand)
        assert cossim_mat.shape[0] == cossim_mat.shape[1] == bs
        # cossim_mat[x][y] == cos(z1[x], z2[y])
        exp_mat = torch.exp(cossim_mat / T)
        loss_line = exp_mat.diagonal() / exp_mat.sum(dim=1)
        loss_line = -torch.log(loss_line)
        loss = loss_line.sum()
        return loss

    def loss(
        self,
        pno_tree_y,
        chd,
        pr_mat,
        feat_y,
        prmat_y,
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
            dist_chd, dist_sym, z_sym, z_y_sym
        ) = self.run(pno_tree_y, chd, pr_mat, feat_y, prmat_y, tfr1, tfr2, tfr3)

        return self.loss_function(
            pno_tree_y, feat_y, chd, recon_pitch, recon_dur, recon_root, recon_chroma,
            recon_bass, recon_feat, dist_chd, dist_sym, z_sym, z_y_sym, beta, weights
        )

    @classmethod
    def init_model(
        cls, z_chd_dim=256, z_sym_dim=256, pt_txtenc_path=None, model_path=None
    ):
        """Fast model initialization."""

        name = 'prvae'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        prmat_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        # contrastive learning
        pt_prmat_y_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        model = cls(
            name, device, chord_enc, chord_dec, prmat_enc, diffusion_nn, feat_dec,
            pianotree_dec, False, pt_prmat_y_enc
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    @classmethod
    def init_model_pretrained_txtenc(
        cls, z_chd_dim=256, z_sym_dim=256, pt_txtenc_path=None, model_path=None
    ):
        """
        Fast model initialization with pretrained texture encoder from Polydis
        """
        name = 'prvae_pttxtenc'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        # load pretrained texture encoder from ziyu's polydis
        prmat_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        # contrastive learning
        pt_prmat_y_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        model = cls(
            name, device, chord_enc, chord_dec, prmat_enc, diffusion_nn, feat_dec,
            pianotree_dec, True, pt_prmat_y_enc
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    def inference(self, chord, pr_mat, use_zx_txt=False):
        """
        Forward path during inference.

        :param chord: (B, 8, 36) chord input
        :param pr_mat: (B, 32, 128) symbolic piano roll matrices.
        :param use_zx_txt: True when using zx_txt (not going through diffusion_nn)
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z_chd = self.chord_enc(chord).mean

            dist_x_sym = self.prmat_enc(pr_mat)
            if use_zx_txt:
                z_sym = dist_x_sym.mean  # this is z_x_sym
            else:
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


class PianoReductionVAE(PytorchModel):
    """
    The proposed piano reduction VAE model without contrastive loss
    """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_chd', 'kl_sym', 'chord_l',
        'root_l', 'chroma_l', 'bass_l', 'feat_l', 'bass_feat_l', 'int_feat_l',
        'rhy_feat_l', 'beta'
    ]

    def __init__(
        self,
        name,
        device,
        chord_enc: ChordEncoder,
        chord_dec: ChordDecoder,
        prmat_enc: TextureEncoder,
        diffusion_nn: NaiveNN,
        feat_dec: FeatDecoder,
        pianotree_dec: PianoTreeDecoder,
        prmat_enc_type,
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

        self.prmat_enc_type = prmat_enc_type
        print(f"prmat_enc_type : {self.prmat_enc_type}")

    @property
    def z_chd_dim(self):
        return self.chord_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    def run(self, pno_tree_y, chd, pr_mat, feat_y, prmat_y, tfr1, tfr2, tfr3):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # chord representation
        try:
            dist_chd = self.chord_enc(chd)
            z_chd = dist_chd.rsample()

            # symbolic-texture representation
            if self.prmat_enc_type == "pretrained":
                with torch.no_grad():
                    dist_x_sym = self.prmat_enc(pr_mat)
            else:
                dist_x_sym = self.prmat_enc(pr_mat)
            # z_x_sym = dist_x_sym.rsample()
            # print(dist_x_sym)

            if self.prmat_enc_type == "finetune":
                # do not use nn in between, simply polydis with feats
                dist_sym = dist_x_sym
                z_sym = dist_sym.rsample()
            else:
                # diffusion model: transform z_x_txt to z_y_txt
                dist_sym = self.diffusion_nn(dist_x_sym)
                z_sym = dist_sym.rsample()  # this is z_y_txt

        except ValueError as e:
            print(e)
            chd = chd.cpu().detach().numpy()
            pr_mat = pr_mat.cpu().detach().numpy()
            retrieve_midi_from_chd([onehot_to_chd(c) for c in chd], "error_chd.mid")
            retrieve_midi_from_prmat(pr_mat, "error_prmat.mid")
            print("error chord retrieved")
            exit(0)

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
        prmat_y,
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
        ) = self.run(pno_tree_y, chd, pr_mat, feat_y, prmat_y, tfr1, tfr2, tfr3)

        return self.loss_function(
            pno_tree_y, feat_y, chd, recon_pitch, recon_dur, recon_root, recon_chroma,
            recon_bass, recon_feat, dist_chd, dist_sym, beta, weights
        )

    @classmethod
    def init_model(
        cls, z_chd_dim=256, z_sym_dim=256, pt_txtenc_path=None, model_path=None
    ):
        """Fast model initialization."""

        name = 'prvae'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        if pt_txtenc_path is not None:
            prmat_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)
        else:
            prmat_enc = TextureEncoder(z_dim=z_sym_dim)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(
            name,
            device,
            chord_enc,
            chord_dec,
            prmat_enc,
            diffusion_nn,
            feat_dec,
            pianotree_dec,
            prmat_enc_type=None
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    @classmethod
    def init_model_pretrained_txtenc(
        cls, z_chd_dim=256, z_sym_dim=256, pt_txtenc_path=None, model_path=None
    ):
        """
        Fast model initialization with pretrained texture encoder from Polydis
        """
        name = 'prvae_pttxtenc'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        # load pretrained texture encoder from ziyu's trained model
        prmat_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(
            name,
            device,
            chord_enc,
            chord_dec,
            prmat_enc,
            diffusion_nn,
            feat_dec,
            pianotree_dec,
            prmat_enc_type="pretrained"
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    @classmethod
    def init_model_finetune_txtenc(
        cls, z_chd_dim=256, z_sym_dim=256, pt_txtenc_path=None, model_path=None
    ):
        """
        Fast model initialization to finetune texture encoder from Polydis
        """
        name = 'finetune_txtenc'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        # load pretrained texture encoder from ziyu's trained model
        prmat_enc = load_pretrained_txt_enc(pt_txtenc_path, z_sym_dim, device)

        diffusion_nn = NaiveNN(z_sym_dim, z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_sym_dim)

        z_pt_dim = z_chd_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(
            name,
            device,
            chord_enc,
            chord_dec,
            prmat_enc,
            diffusion_nn,
            feat_dec,
            pianotree_dec,
            prmat_enc_type="finetune"
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    def inference(self, chord, pr_mat):
        """
        Forward path during inference.

        :param chord: (B, 8, 36) chord input
        :param pr_mat: (B, 32, 128) symbolic piano roll matrices.
        :param use_zx_txt: True when using zx_txt (not going through diffusion_nn,
                for debugging)
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """
        print("inferencing no contrastive...")

        self.eval()
        with torch.no_grad():
            z_chd = self.chord_enc(chord).mean

            dist_x_sym = self.prmat_enc(pr_mat)
            if self.prmat_enc_type == "finetune":
                print("infer finetune model: using zx_txt...")
                # pt_prmat_enc_tmp = load_pretrained_txt_enc(
                #     "data/Polydis_pretrained/model_master_final.pt", 256, self.device
                # )
                # dist_x_sym = pt_prmat_enc_tmp(pr_mat)
                z_sym = dist_x_sym.mean  # this is z_x_sym
            else:
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
