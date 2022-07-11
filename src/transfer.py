import os
import pickle
import sys
import numpy as np
from config import prepare_model
from dirs import *
from dataset import DataSampleNpz
from data_loader import create_data_loaders
from utils import estx_to_midi_file
import torch
import pretty_midi as pm
from datetime import datetime


def model_compute(model, fname, device):
    """Batching the input and call model.inference()."""

    song = DataSampleNpz(fname, load_chord=True)
    _, chd_x, prmat_x, _ = song.get_whole_song_data()
    print(chd_x.shape, prmat_x.shape)

    predictions = model.inference(chd_x.to(device), prmat_x.to(device))
    return predictions


if __name__ == "__main__":
    model_path = sys.argv[1]

    split_fpath = os.path.join(TRAIN_SPLIT_DIR, "split_dict.pickle")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(split_fpath, "rb") as f:
        split = pickle.load(f)

    print(split[(None, None)][1])
    num = int(input("choose one:"))
    test = split[(None, None)][1][num]
    # test_fpath = os.path.join(QUANTIZED_DATA_DIR, test)
    print(test)

    if len(sys.argv) > 2:
        output_fpath = sys.argv[2]
    else:
        output_fpath = f"exp/inference_[{test}]_{datetime.now().strftime('%m-%d_%H:%M:%S')}.mid"

    model = prepare_model("prvae", model_path=model_path)
    predictions = model_compute(model, test, device)
    estx_to_midi_file(predictions, output_fpath)
