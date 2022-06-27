import muspy
import os
import numpy as np
import mir_eval
import csv
from tqdm import tqdm
from os.path import join

# duration of one beat
one_beat = 0.5
bin = 4


def get_note_matrix(fpath, fpath_test):
    """
    get note matrix: same format as pop909-4-bin
    plus an instrument program num appended at the end
    """
    # music = pm.PrettyMIDI(fpath)
    music = muspy.read_midi(fpath)

    # adjust resolution
    music.adjust_resolution(bin)
    # music.write_midi(join(fpath_test))

    notes = []
    for inst in music.tracks:
        for note in inst.notes:
            # start_beat = int(note.start / one_beat)
            # start_bin = int((note.start - start_beat * one_beat) / (one_beat / bin))
            start_beat = int(note.time / bin)
            start_bin = note.time - start_beat * bin
            # end_beat = int(note.end / one_beat)
            # end_bin = int((note.end - end_beat * one_beat) / (one_beat / bin))
            # duration = int((note.end - note.start) / (one_beat / bin))
            duration = note.duration
            if duration > 0:
                notes.append(
                    [
                        start_beat,
                        start_bin,
                        duration,
                        note.pitch,
                        note.velocity,
                        inst.program,
                    ]
                )
    notes.sort(key=lambda x: x[0] * bin + x[1])
    return notes


def get_chord_matrix(fpath):
    """
    chord matrix [M * 14], each line represent the chord of a beat
    same format as mir_eval.chord.encode():
        root_number(1), semitone_bitmap(12), bass_number(1)
    """
    file = csv.reader(open(fpath), delimiter="\t")
    beat_cnt = 0
    chords = []
    for line in file:
        start = float(line[0])
        end = float(line[1])
        chord = line[2]
        assert ((end - start) / one_beat).is_integer()
        beat_num = int((end - start) / one_beat)
        for _ in range(beat_num):
            beat_cnt += 1
            chords.append(mir_eval.chord.encode(chord))
    return chords


if __name__ == "__main__":
    dpath = "data/LOP_aligned"
    dpath_chd = "data/LOP_chord"

    dpath_test = "data/LOP_4_bin"
    dpath_save = "data/LOP_4_bin"
    os.system(f"rm -rf {dpath_save}")

    for piece in tqdm(os.listdir(dpath)):
        os.system(f"mkdir -p {join(dpath_save, piece)}")
        for ver in os.listdir(join(dpath, piece)):
            fpath = join(dpath, piece, ver)
            note_mat = get_note_matrix(fpath, join(dpath_test, piece, ver))
            # np.save(join(dpath_save, piece, ver[:-4]), note_mat)

            fpath_chd = join(dpath_chd, piece, ver[:-4]) + ".out"
            chord_mat = get_chord_matrix(fpath_chd)

            np.savez(join(dpath_save, piece, ver[:-4]), note=note_mat, chord=chord_mat)
