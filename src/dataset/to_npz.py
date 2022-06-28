import muspy
import pretty_midi as pm
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
            onset_beat = int(note.time / bin)
            onset_bin = note.time - onset_beat * bin
            # end_beat = int(note.end / one_beat)
            # end_bin = int((note.end - end_beat * one_beat) / (one_beat / bin))
            # duration = int((note.end - note.start) / (one_beat / bin))
            duration = note.duration
            if duration > 0:
                notes.append(
                    [
                        onset_beat,
                        onset_bin,
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
            # see https://craffel.github.io/mir_eval/#mir_eval.chord.encode
            chd_enc = mir_eval.chord.encode(chord)

            root = chd_enc[0]
            # make chroma and bass absolute
            chroma_bitmap = chd_enc[1]
            chroma_bitmap = np.roll(chroma_bitmap, root)
            bass = (chd_enc[2] + root) % 12

            line = [root]
            for _ in chroma_bitmap:
                line.append(_)
            line.append(bass)

            chords.append(line)
    return chords


def get_start_table(notes):
    """
    i-th row indicates the starting row of the "notes" array at i-th beat.
    """
    total_beat = notes[-1][0]
    row_cnt = 0
    start_table = []
    for beat in range(total_beat):
        while notes[row_cnt][0] < beat:
            row_cnt += 1
        start_table.append(row_cnt)
    return start_table


def get_downbeat_position(fpath):
    """
    simply get the downbeat position of the given midi file
    """
    music = pm.PrettyMIDI(fpath)
    return [int(b / one_beat) for b in music.get_downbeats()]


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

            start_table = get_start_table(note_mat)

            db_pos = get_downbeat_position(fpath)

            np.savez(
                join(dpath_save, piece, ver[:-4]),
                notes=note_mat,
                chord=chord_mat,
                start_table=start_table,
                db_pos=db_pos,
            )
