import sys
import os

# sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from models import PianoReductionVAE
from dataset import PianoOrchDataset
from data_loader import PianoOrchDataLoader, create_data_loaders
from constants import AUG_P
from dirs import RESULT_PATH
from train import train_model

PR_CONFIG = {'z_chd_dim': 256, 'z_sym_dim': 256}
PR_PTTXTENC_CONFIG = {
    'z_chd_dim': 256,
    'z_sym_dim': 256,
    'pretrained_path': 'data/Polydis_pretrained/model_master_final.pt'
}

NOCHD_CONFIG = {'z_aud_dim': 320, 'z_sym_dim': 192}

SUPERVISED_CONFIG = {'z_dim': 512}

# Ensuring path is valid when importing from outside the project.
PARAM_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'params'))
TRANSCRIBER_PATH = os.path.join(PARAM_PATH, 'pretrained_onsets_and_frames.pt')

TRAIN_CONFIG = {
    'batch_size': 64,
    'num_workers': 0,
    'meter': 2,
    'n_subdiv': 2,
    'parallel': False,
    'load_data_at_start': False
}

LR = 5e-5
BETA = 0.5
CORRUPT = {0: False, 1: True, 2: True, 3: False}
AUTOREGRESSIVE = {0: False, 1: False, 2: False, 3: True}
N_EPOCH = 100


def prepare_model(model_id, model_path=None):
    if model_id == 'prvae':
        model = PianoReductionVAE.init_model(
            z_chd_dim=PR_CONFIG['z_chd_dim'],
            z_sym_dim=PR_CONFIG['z_sym_dim'],
            model_path=model_path
        )
    elif model_id == "prvae_pttxtenc":
        model = PianoReductionVAE.init_model_pretrained_txtenc(
            z_chd_dim=PR_PTTXTENC_CONFIG['z_chd_dim'],
            z_sym_dim=PR_PTTXTENC_CONFIG['z_sym_dim'],
            pretrained_path=PR_PTTXTENC_CONFIG['pretrained_path'],
            model_path=model_path
        )
    else:
        raise NotImplementedError
    return model


def prepare_data_loaders(test_mode):
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

    return create_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=TRAIN_CONFIG['num_workers'],
        meter=TRAIN_CONFIG['meter'],
        n_subdiv=TRAIN_CONFIG['n_subdiv'],
    )


def result_path_folder_path(model_id):
    folder_name = model_id
    folder_path = os.path.join(RESULT_PATH, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


class TrainingCall:
    def __init__(self, model_id):
        assert model_id in ["prvae", "prvae_pttxtenc"]

        self.model_id = model_id
        self.result_path = result_path_folder_path(model_id)

    def __call__(self, test_mode, model_path, run_epochs, readme_fn):
        model = prepare_model(self.model_id, model_path)
        data_loaders = prepare_data_loaders(test_mode)
        train_model(
            model=model,
            data_loaders=data_loaders,
            readme_fn=readme_fn,
            n_epoch=N_EPOCH,
            parallel=TRAIN_CONFIG['parallel'],
            lr=LR,
            writer_names=model.writer_names,
            load_data_at_start=TRAIN_CONFIG['load_data_at_start'],
            beta=BETA,
            run_epochs=run_epochs,
            result_path=self.result_path
        )
