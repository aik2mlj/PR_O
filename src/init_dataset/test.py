import numpy as np
import mir_eval as me
import pretty_midi as pm
import sys

if __name__ == "__main__":
    fpath = sys.argv[1]

    data = np.load(fpath, allow_pickle=True)
    n = data["notes"]
    c = data["chord"]
    np.set_printoptions(threshold=np.inf)
