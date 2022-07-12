import pickle
import os
from sklearn.model_selection import train_test_split


def save_dict(path, dict_file):
    with open(path, "wb") as handle:
        pickle.dump(dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


QUANTIZED_DATA_DIR = "data/LOP_4_bin"
TRAIN_SPLIT_DIR = "data/train_split"

if __name__ == "__main__":
    dnames = os.listdir(QUANTIZED_DATA_DIR)
    train_d, val_d = train_test_split(dnames, test_size=0.1)
    dict = {}
    dict[(None, None)] = (train_d, val_d)
    save_dict(os.path.join(TRAIN_SPLIT_DIR, "split_dict.pickle"), dict)
