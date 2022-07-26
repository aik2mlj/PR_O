import sys
import os

# sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from models import PianoReductionVAE, PianoReductionVAE_contrastive
from model_pnotree import PianoTree_PRVAE
from dataset import PianoOrchDataset
from data_loader import PianoOrchDataLoader, create_data_loaders
from polydis_dataset.dataset_polydis import AudioMidiDataLoaders, create_data_loaders_polydis
from constants import AUG_P
from dirs import RESULT_PATH
from train import train_model

PR_CONFIG = {
    'z_chd_dim': 256,
    'z_sym_dim': 256,
    'pt_polydis_path': 'data/Polydis_pretrained/model_master_final.pt'
}
PR_PTTXTENC_CONFIG = {
    'z_chd_dim': 256,
    'z_sym_dim': 256,
    'pt_polydis_path': 'data/Polydis_pretrained/model_master_final.pt'
}
PNOTREE_PR_CONFIG = {"z_dim": 512, "pt_pianotree_path": "TODO"}

# SUPERVISED_CONFIG = {'z_dim': 512}

TRAIN_CONFIG = {
    'batch_size': 128,
    'num_workers': 0,
    'meter': 2,
    'n_subdiv': 2,
    'parallel': False,
    'load_data_at_start': False,
    'lr': 1e-3,
    'beta': 0.1,
    'n_epoch': 50,
}

FINETUNE_CONFIG = {
    'batch_size': 128,
    'num_workers': 0,
    'meter': 2,
    'n_subdiv': 2,
    'parallel': False,
    'load_data_at_start': False,
    'lr': 1e-3,
    'beta': 0.1,
    'n_epoch': 30,
}

# LR = 1e-3
# BETA = 0.1
# N_EPOCH = 50


def prepare_model(model_id, model_path=None):
    if model_id == 'prvae':
        model = PianoReductionVAE.init_model(
            z_chd_dim=PR_CONFIG['z_chd_dim'],
            z_sym_dim=PR_CONFIG['z_sym_dim'],
            pt_txtenc_path=PR_CONFIG['pt_polydis_path'],
            model_path=model_path
        )
    elif model_id == "prvae_pttxtenc":
        model = PianoReductionVAE.init_model_pretrained_txtenc(
            z_chd_dim=PR_PTTXTENC_CONFIG['z_chd_dim'],
            z_sym_dim=PR_PTTXTENC_CONFIG['z_sym_dim'],
            pt_txtenc_path=PR_PTTXTENC_CONFIG['pt_polydis_path'],
            model_path=model_path
        )
    elif model_id == "prvae_contra":
        model = PianoReductionVAE_contrastive.init_model(
            z_chd_dim=PR_CONFIG['z_chd_dim'],
            z_sym_dim=PR_CONFIG['z_sym_dim'],
            pt_txtenc_path=PR_CONFIG['pt_polydis_path'],
            model_path=model_path
        )
    elif model_id == "prvae_pttxtenc_contra":
        model = PianoReductionVAE_contrastive.init_model_pretrained_txtenc(
            z_chd_dim=PR_PTTXTENC_CONFIG['z_chd_dim'],
            z_sym_dim=PR_PTTXTENC_CONFIG['z_sym_dim'],
            pt_txtenc_path=PR_PTTXTENC_CONFIG['pt_polydis_path'],
            model_path=model_path
        )
    elif model_id == "finetune_txtenc":
        model = PianoReductionVAE.init_model_finetune_txtenc(
            z_chd_dim=PR_CONFIG['z_chd_dim'],
            z_sym_dim=PR_CONFIG['z_sym_dim'],
            pt_txtenc_path=PR_CONFIG['pt_polydis_path'],
            model_path=model_path
        )
    elif model_id == "pianotree_prvae":
        model = PianoTree_PRVAE.init_model(
            z_dim=PNOTREE_PR_CONFIG['z_dim'],
            pt_enc_path=PNOTREE_PR_CONFIG['pt_pianotree_path'],
            model_path=model_path
        )
    else:
        raise NotImplementedError
    return model


def prepare_data_loaders(test_mode, model_id, dataset_id):
    if test_mode:
        tv_song_paths = (["bouliane-0", "bouliane-1", "bouliane-2"], ["bouliane-3"])
        train_set, valid_set = PianoOrchDataset.load_with_train_valid_paths(
            tv_song_paths
        )
        batch_size = 16
        aug_p = AUG_P / AUG_P.sum()
        return PianoOrchDataLoader.get_loaders(
            batch_size, batch_size, train_set, valid_set, True, False, aug_p,
            TRAIN_CONFIG['num_workers']
        )

    if dataset_id == "pr_o":
        if model_id == "finetune_txtenc":
            return create_data_loaders(
                batch_size=FINETUNE_CONFIG['batch_size'],
                num_workers=FINETUNE_CONFIG['num_workers'],
                meter=FINETUNE_CONFIG['meter'],
                n_subdiv=FINETUNE_CONFIG['n_subdiv'],
                type="all_x"
            )
        elif model_id == "pianotree_prvae":
            return create_data_loaders(
                batch_size=TRAIN_CONFIG['batch_size'],
                num_workers=TRAIN_CONFIG['num_workers'],
                meter=TRAIN_CONFIG['meter'],
                n_subdiv=TRAIN_CONFIG['n_subdiv'],
                type="pianotree"
            )
        else:
            return create_data_loaders(
                batch_size=TRAIN_CONFIG['batch_size'],
                num_workers=TRAIN_CONFIG['num_workers'],
                meter=TRAIN_CONFIG['meter'],
                n_subdiv=TRAIN_CONFIG['n_subdiv'],
                type=None
            )

    elif dataset_id == "polydis":
        return create_data_loaders_polydis(
            batch_size=TRAIN_CONFIG['batch_size'],
            num_workers=TRAIN_CONFIG['num_workers'],
            meter=TRAIN_CONFIG['meter'],
            n_subdiv=TRAIN_CONFIG['n_subdiv'],
        )
    else:
        raise RuntimeError


def result_path_folder_path(model_id):
    folder_name = model_id
    folder_path = os.path.join(RESULT_PATH, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


class TrainingCall:
    def __init__(self, model_id, dataset_id="pr_o"):
        assert model_id in [
            "prvae", "prvae_pttxtenc", "prvae_contra", "prvae_pttxtenc_contra",
            "finetune_txtenc"
        ]
        assert dataset_id in ["pr_o", "polydis"]

        self.model_id = model_id
        self.dataset_id = dataset_id
        self.result_path = result_path_folder_path(
            model_id + "+" + dataset_id if dataset_id == "polydis" else model_id
        )

        # print training setting
        print("====== TrainingCall info:")
        print(f"model: {model_id}")
        if "pttxtenc" in model_id:
            print(f"prvae_pttxtenc: {PR_PTTXTENC_CONFIG}")
        elif "pianotree" in model_id:
            print(f"pianotree_prvae: {PNOTREE_PR_CONFIG}")
        else:
            print(f"prvae: {PR_CONFIG}")
        print(f"dataset_id: {dataset_id}")
        print(f"result_path: {self.result_path}")
        if model_id == "finetune_txtenc":
            print(f"finetune_config: {FINETUNE_CONFIG}")
        else:
            print(f"train_config: {TRAIN_CONFIG}")
        print("======")

    def __call__(self, test_mode, model_path, run_epochs, readme_fn):
        model = prepare_model(self.model_id, model_path)
        data_loaders = prepare_data_loaders(test_mode, self.model_id, self.dataset_id)
        if self.model_id == "finetune_txtenc":
            train_model(
                model=model,
                data_loaders=data_loaders,
                readme_fn=readme_fn,
                n_epoch=FINETUNE_CONFIG['n_epoch'],
                parallel=FINETUNE_CONFIG['parallel'],
                lr=FINETUNE_CONFIG['lr'],
                writer_names=model.writer_names,
                load_data_at_start=FINETUNE_CONFIG['load_data_at_start'],
                beta=FINETUNE_CONFIG['beta'],
                run_epochs=run_epochs,
                result_path=self.result_path
            )
        else:
            train_model(
                model=model,
                data_loaders=data_loaders,
                readme_fn=readme_fn,
                n_epoch=TRAIN_CONFIG['n_epoch'],
                parallel=TRAIN_CONFIG['parallel'],
                lr=TRAIN_CONFIG['lr'],
                writer_names=model.writer_names,
                load_data_at_start=TRAIN_CONFIG['load_data_at_start'],
                beta=TRAIN_CONFIG['beta'],
                run_epochs=run_epochs,
                result_path=self.result_path
            )
