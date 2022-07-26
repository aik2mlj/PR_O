import os
import pickle
import sys
import numpy as np
from config import prepare_model
from dirs import *
from dataset import DataSampleNpz
from polydis_dataset.data_sample_polydis import TrainDataSample as PolydisDataSample
from utils import estx_to_midi_file
import torch
import pretty_midi as pm
from datetime import datetime

# model_name = "prvae"


def model_compute(model_name, dataset_name, fname, device):
    """Batching the input and call model.inference()."""

    model = prepare_model(model_name, model_path=model_path)
    if dataset_name == "pr_o":
        song = DataSampleNpz(fname, load_chord=True)
        _, chd_x, prmat_x, _, _ = song.get_whole_song_data()
        print(chd_x.shape, prmat_x.shape)
    else:
        assert dataset_name == "polydis"
        song = PolydisDataSample(fname)
        _, chd_x, prmat_x, _ = song.get_whole_song_data()
        print(chd_x.shape, prmat_x.shape)

    predictions = model.inference(chd_x.to(device), prmat_x.to(device))
    return predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = sys.argv[1]
    split_name = model_path.split("/")[1].split("+")
    model_name = split_name[0]
    dataset_name = "pr_o" if len(split_name) < 2 else split_name[1]
    print(f"model_name: {model_name}")
    print(f"dataset_name: {dataset_name}")

    if dataset_name == "pr_o":
        split_fpath = os.path.join(TRAIN_SPLIT_DIR, "split_dict.pickle")
        with open(split_fpath, "rb") as f:
            split = pickle.load(f)
        print(split[(None, None)][1])
        num = int(input("choose one:"))
        test = split[(None, None)][1][num]
    elif dataset_name == "polydis":
        split_fpath = os.path.join(POLYDIS_TRAIN_SPLIT_DIR, "split_dict.pickle")
        with open(split_fpath, "rb") as f:
            split = pickle.load(f)
        print(split[(2, 2)][1])
        num = int(input("choose one:"))
        test = num
    else:
        print("dataset_name unmatched!")
        exit(1)

    # test_fpath = os.path.join(QUANTIZED_DATA_DIR, test)
    print(test)

    if len(sys.argv) > 2:
        output_fpath = sys.argv[2]
    else:
        output_fpath = f"exp/inference_[{test}]_{datetime.now().strftime('%m-%d_%H:%M:%S')}.mid"

    predictions = model_compute(model_name, dataset_name, test, device)
    estx_to_midi_file(predictions, output_fpath)
