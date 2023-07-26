from pathlib import Path
from tqdm import tqdm

from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct

def merge_midi_helper(filepath: Path, track_num: int, inst_list: list[int]=['kick','snare','hh','bass','piano']) -> parser.MidiFile:
    """Helper function for merge_midi"""

    # instruments and their programs per General MIDI 1
    # https://www.pgmusic.com/tutorial_gm.htm
    instruments_dict = {'kick':0,'snare':0,'hh':0,'bass':38,'piano':0}
    instruments_order = {'kick':0,'snare':1,'hh':2,'bass':3,'piano':4}

    # instruments to actually merge
    instruments = {}
    inst_list.sort(key=lambda x: instruments_order[x])
    instruments = {x: instruments_dict[x] for x in inst_list}

    # create merged MIDI and take ticks_per_beat of kick
    tpb_kick = parser.MidiFile(filepath.joinpath(f"{track_num}-kick.mid")).ticks_per_beat
    merged_midi = parser.MidiFile(ticks_per_beat=tpb_kick)\

    # create the drum instrument
    merged_midi.instruments.append(ct.Instrument(program=0, is_drum=True, name='drums'))

    # merge
    for inst, prog in tqdm(instruments.items(), desc=f"Merging Tracks"):
        filename = filepath.joinpath(f"{track_num}-{inst}.mid")

        # parse individual stem track
        track = parser.MidiFile(filename)

        # check ticks per beat
        if track.ticks_per_beat != merged_midi.ticks_per_beat:
            raise ValueError("Ticks per beat of individual tracks don't match")

        # merge drums first
        if inst in ['kick', 'snare', 'hh']:
            merged_midi.instruments[0].notes += track.instruments[0].notes

        else:
            # if not drum, just add names and programs
            track.instruments[0].name = inst
            track.instruments[0].program = prog
            merged_midi.instruments.append(track.instruments[0])

    # sort merged drum notes by start
    merged_midi.instruments[0].notes.sort(key=lambda x:x.start)

    return merged_midi

def merge_midi(sample_size: int=1, inst_list: list[int]=['kick','snare','hh','bass','piano'], filepath: str='', add_filename: str=""):
    """Merges tracks into single Type 1 mid file
    Filepath = folder with 'dataset' folder inside
    Filenames will be <sample_no><"-" + additional_filename>.mid"""

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError("Please check filepath")
    path = path.joinpath('dataset')

    if add_filename != "":
        add_filename = "-" + add_filename

    for i in tqdm(range(sample_size), desc="Loading Track..."):
        # Path object with individual track folders
        trackpath = path.joinpath(str(i))
        merged_midi = merge_midi_helper(trackpath, i, inst_list)
        merged_midi.dump(path.joinpath(f"{i}{add_filename}.mid"))

if __name__ == "__main__":
    
    sample_size = 29
    merge_midi(sample_size=sample_size)