import muspy
import sys
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


def get_note_matrix(fpath):
    """
    get note matrix: same format as pop909-4-bin
        for piano, this function simply extracts notes
        for orchestra, this function "flattens" the notes into one single track
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
    # sort according to (start, duration)
    notes.sort(key=lambda x: (x[0] * bin + x[1], x[2]))
    return notes


def dedup_note_matrix(notes):
    """
    remove duplicated notes (because of multiple tracks)
    """

    last = []
    notes_dedup = []
    for i, note in enumerate(notes):
        if i != 0:
            if note[:4] != last[:4]:
                # if start, duration and pitch are not the same
                notes_dedup.append(note)
        else:
            notes_dedup.append(note)
        last = note
    print(f"dedup: {len(notes) - len(notes_dedup)} : {len(notes)}")

    return notes_dedup


def retrieve_midi_from_nmat(notes, output_fpath):
    """
    retrieve midi from note matrix
    """
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    for note in notes:
        onset_beat, onset_bin, duration, pitch, velocity, program = note
        start = onset_beat * one_beat + onset_bin * one_beat / float(bin)
        end = start + duration * one_beat / float(bin)
        pm_note = pm.Note(velocity, pitch, start, end)
        piano.notes.append(pm_note)

    midi.instruments.append(piano)
    midi.write(output_fpath)


def get_chord_matrix(fpath):
    """
    chord matrix [M * 14], each line represent the chord of a beat
    same format as mir_eval.chord.encode():
        root_number(1), semitone_bitmap(12), bass_number(1)
    inputs are generated from junyan's algorithm
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


def retrieve_midi_from_chd(chords, output_fpath):
    """
    retrieve midi from chords
    """
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    for beat, chord in enumerate(chords):
        root = chord[0]
        chroma = chord[1:13]
        bass = chord[13]

        chroma = np.roll(chroma, -bass)
        c3 = 48
        for i, n in enumerate(chroma):
            if n == 1:
                note = pm.Note(
                    velocity=80,
                    pitch=c3 + i + bass,
                    start=beat * one_beat,
                    end=(beat + 1) * one_beat
                )
                piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_fpath)


def get_start_table(music, notes):
    """
    i-th row indicates the starting row of the "notes" array at i-th beat.
    """

    # simply add 8-beat padding in case of out-of-range index
    total_beat = int(music.get_end_time() / one_beat) + 8
    row_cnt = 0
    start_table = []
    for beat in range(total_beat):
        while row_cnt < len(notes) and notes[row_cnt][0] < beat:
            row_cnt += 1
        start_table.append(row_cnt)

    return start_table


def get_downbeat_position(music):
    """
    simply get the downbeat position of the given midi file
    """
    db_pos = [int(b / one_beat) for b in music.get_downbeats()]
    return db_pos


def make_npz_for_single_midi(fpath, fpath_chd, output_fpath):
    note_mat = get_note_matrix(fpath)
    # np.save(join(dpath_save, piece, ver[:-4]), note_mat)

    chord_mat = get_chord_matrix(fpath_chd)

    music = pm.PrettyMIDI(fpath)
    start_table = get_start_table(music, note_mat)
    db_pos = get_downbeat_position(music)

    np.savez(
        output_fpath,
        notes=note_mat,
        chord=chord_mat,
        start_table=start_table,
        db_pos=db_pos,
    )


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
            dpath_s_dir = join(dpath_save, piece)

            note_mat = get_note_matrix(fpath)
            note_mat = dedup_note_matrix(note_mat)
            # np.save(join(dpath_save, piece, ver[:-4]), note_mat)
            retrieve_midi_from_nmat(
                note_mat, join(dpath_s_dir, ver[:-4] + "_flated.mid")
            )

            fpath_chd = join(dpath_chd, piece, ver[:-4]) + ".out"
            os.system(f"cp {fpath_chd} {dpath_s_dir}/")
            chord_mat = get_chord_matrix(fpath_chd)

            retrieve_midi_from_chd(chord_mat, join(dpath_s_dir, ver[:-4] + "_chd.mid"))

            music = pm.PrettyMIDI(fpath)
            start_table = get_start_table(music, note_mat)

            db_pos = get_downbeat_position(music)

            np.savez(
                join(dpath_s_dir, ver[:-4]),
                notes=note_mat,
                chord=chord_mat,
                start_table=start_table,
                db_pos=db_pos,
            )
