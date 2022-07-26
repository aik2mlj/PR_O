from amc_dl.torch_plus import PytorchModel
from amc_dl.torch_plus.train_utils import get_zs_from_dists, kl_with_normal
import torch
from torch import nn
from torch.distributions import Normal
from dl_modules import (
    ChordEncoder, ChordDecoder, PianoTreeDecoder, TextureEncoder, FeatDecoder, NaiveNN,
    PianoTreeEncoder
)
from utils import (
    load_pretrained_pianotree_enc, retrieve_midi_from_chd, estx_to_midi_file,
    onehot_to_chd, retrieve_midi_from_prmat
)

# TODO: double check x and y


class PianoTree_PRVAE(PytorchModel):
    """
    The proposed piano reduction VAE model without contrastive loss
    """

    writer_names = ['loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_x', 'beta']

    def __init__(
        self,
        name,
        device,
        pianotree_enc: PianoTreeEncoder,
        diffusion_nn: NaiveNN,
        # feat_dec: FeatDecoder,
        pianotree_dec: PianoTreeDecoder,
        enc_type,
    ):

        super(PianoTree_PRVAE, self).__init__(name, device)

        self.pianotree_enc = pianotree_enc

        # pianotree encoder + diffusion model = symbolic encoder
        self.diffusion_nn = diffusion_nn  # TODO

        # feat_dec + pianotree_dec = symbolic decoder
        # self.feat_dec = feat_dec
        # self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = pianotree_dec

        self.sym_enc_type = enc_type

    @property
    def z_sym_dim(self):
        return self.pianotree_enc.z_size

    def run(self, pno_tree_x, pno_tree_y, tfr1, tfr2, tfr3):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # symbolic-texture representation
        if self.sym_enc_type == "pretrained":
            with torch.no_grad():
                dist_x, emb_x = self.pianotree_enc(pno_tree_x)
        else:
            dist_x, emb_x = self.pianotree_enc(pno_tree_x)
        # z_x_sym = dist_x_sym.rsample()
        # print(dist_x_sym)

        z_x = dist_x.rsample()
        if self.sym_enc_type == "finetune":
            # do not use nn in between, simply polydis with feats
            z = z_x
        else:
            # diffusion model: transform z_x_txt to z_y_txt
            z = self.diffusion_nn(z_x)  # this is z_y_txt

        # reconstruction of symbolic feature from z_y_txt
        # recon_feat = self.feat_dec(z, False, tfr1, feat_y)

        # embed the reconstructed feature (without applying argmax)
        # feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree_y)

        # pianotree decoder
        recon_pitch, recon_dur = self.pianotree_dec(
            z, False, embedded_pno_tree, pno_tree_lgths, tfr1, tfr2, None
        )

        return (recon_pitch, recon_dur, dist_x)

    def loss_function(self, pno_tree_y, recon_pitch, recon_dur, dist_x, beta, weights):
        """Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = self.pianotree_dec.recon_loss(
            pno_tree_y, recon_pitch, recon_dur, weights, False
        )

        # kl losses
        kl_x = kl_with_normal(dist_x)
        kl_l = beta * (kl_x)

        # TODO: contrastive loss

        loss = pno_tree_l + kl_l

        return (loss, pno_tree_l, pitch_l, dur_l, kl_l, kl_x, beta)

    def loss(
        self, pno_tree_x, pno_tree_y, tfr1, tfr2, tfr3, beta=0.1, weights=(1, 0.5)
    ):
        """
        Forward path during training with loss computation.

        :param pno_tree_x: (B, 32, 16, 6) orchestra
        :param pno_tree_y: (B, 32, 16, 6) ground truth piano for teacher forcing
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: teacher forcing ratio 3 (for chord decoder)
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        (recon_pitch, recon_dur,
         dist_x) = self.run(pno_tree_x, pno_tree_y, tfr1, tfr2, tfr3)

        return self.loss_function(
            pno_tree_y, recon_pitch, recon_dur, dist_x, beta, weights
        )

    @classmethod
    def init_model(cls, z_dim=512, pt_enc_path=None, model_path=None):
        """Fast model initialization."""

        name = 'pianotree_prvae'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if pt_enc_path is not None:
            pianotree_enc = load_pretrained_pianotree_enc(pt_enc_path, z_dim, device)
        else:
            pianotree_enc = PianoTreeEncoder(device=device, z_size=z_dim)

        diffusion_nn = NaiveNN(z_dim, z_dim)

        pianotree_dec = PianoTreeDecoder(z_size=z_dim, feat_emb_dim=0)

        model = cls(
            name, device, pianotree_enc, diffusion_nn, pianotree_dec, enc_type=None
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    @classmethod
    def init_model_pretrained_enc(cls, z_dim=512, pt_enc_path=None, model_path=None):
        """Fast model initialization."""

        name = 'pianotree_prvae_ptenc'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert pt_enc_path is not None
        pianotree_enc = load_pretrained_pianotree_enc(pt_enc_path, z_dim, device)

        diffusion_nn = NaiveNN(z_dim, z_dim)

        pianotree_dec = PianoTreeDecoder(z_size=z_dim, feat_emb_dim=0)

        model = cls(
            name,
            device,
            pianotree_enc,
            diffusion_nn,
            pianotree_dec,
            enc_type="pretrained"
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    @classmethod
    def init_model_finetune_enc(cls, z_dim=512, pt_enc_path=None, model_path=None):
        """Fast model initialization."""

        name = 'pianotree_prvae_ptenc'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert pt_enc_path is not None
        pianotree_enc = load_pretrained_pianotree_enc(pt_enc_path, z_dim, device)

        diffusion_nn = NaiveNN(z_dim, z_dim)

        pianotree_dec = PianoTreeDecoder(z_size=z_dim, feat_emb_dim=0)

        model = cls(
            name,
            device,
            pianotree_enc,
            diffusion_nn,
            pianotree_dec,
            enc_type="finetune"
        ).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)

        print(f"init_model: {name}")
        return model

    def inference(self, pno_tree_x):
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
            dist_x, emb_x = self.pianotree_enc(pno_tree_x)
            z_x = dist_x.rsample()
            if self.enc_type == "finetune":
                print("infer finetune model: using z_x...")
                z = z_x
            else:
                z = self.diffusion_nn(z_x)

            recon_pitch, recon_dur = self.pianotree_dec(
                z, True, None, None, 0., 0., None
            )

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(), recon_dur.cpu())[0]
        return pred
