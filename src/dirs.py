import os

# dataset paths
QUANTIZED_DATA_DIR = "data/LOP_4_bin"
TRAIN_SPLIT_DIR = "data/train_split"

# polydis data
POLYDIS_DATA_DIR = "data/polydis/quantized_pop909_4_bin"
POLYDIS_TRAIN_SPLIT_DIR = "data/polydis/train_split"

# the path to store demo.
DEMO_FOLDER = "./demo"

# the path to save trained model params and tensorboard log.
RESULT_PATH = "./result"

if not os.path.exists(DEMO_FOLDER):
    os.mkdir(DEMO_FOLDER)

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
