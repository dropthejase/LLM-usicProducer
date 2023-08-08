import argparse
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Union

from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct

def merge_midi_helper(midifiles: list[Union[str, Path]]) -> parser.MidiFile:
    """
    Merges separate drums, bass and piano .mid files into a single Type 1 .mid file
        midifiles:
            - can be list of Paths or str paths to the drums.mid, bass.mid and piano.mid files (but must be in aforementioned order); or
            - a Path or str to the folder containing the individual .mid stems (note they must also be labelled 'drums.mid', 'bass.mid', 'piano.mid')
    """

    # instruments and their programs per General MIDI 1
    # https://www.pgmusic.com/tutorial_gm.htm
    inst_order = {'drums':0,'bass':1,'piano':2}

    # create merged MIDI and take ticks_per_beat of drums
    tpb = parser.MidiFile(midifiles[0]).ticks_per_beat

    # create the merged MidiFile
    merged_midi = parser.MidiFile(ticks_per_beat=tpb)
    merged_midi.instruments.append(ct.Instrument(program=0, is_drum=True, name='drums'))
    merged_midi.instruments.append(ct.Instrument(program=38, is_drum=False, name='bass'))
    merged_midi.instruments.append(ct.Instrument(program=0, is_drum=False, name='piano'))

    ### merge ###
    if len(midifiles) > 3:
        raise ValueError(f"List of midifiles must only comprise [path/to/drums.mid, path/to/bass.mid, path/to/piano.mid]. Got {midifiles} instead.")
    
    for ins in tqdm(midifiles):

        ins = Path(ins) if not isinstance(ins, Path) else ins
        if not ins.exists():
            raise FileNotFoundError(f"{ins} not found")

        track = parser.MidiFile(ins)

        # check ticks per beat
        if track.ticks_per_beat != merged_midi.ticks_per_beat:
            raise ValueError("Ticks per beat of individual tracks don't match")
        
        for k in inst_order.keys():
            if k in ins.stem:
                inst_idx = inst_order[k]
        
        merged_midi.instruments[inst_idx].notes += track.instruments[0].notes


    return merged_midi


def merge_midi(midifiles: Union[str,Path,list[Union[str, Path]]], out_dir: Union[str, Path], merged_filename: str="merged.mid"):
    """
    Merges a list of paths to the 'drums.mid', 'bass.mid' and 'piano.mid' files
        midifiles: a list of paths to the individual instrument stems
        out_dir: where to save the merged .mid file
        merged_filename: name of merged .mid file (make sure to include .mid file extension)
    """

    # if list of paths e.g. ['drums.mid', 'bass.mid', 'piano.mid']
    merged_track = merge_midi_helper(midifiles=midifiles)
    merged_track.dump(f"{out_dir}/{merged_filename}")

def main():

    argparser = argparse.ArgumentParser(prog="Merge Midi", 
                                        description="To merge individual .mid stems to a single multi-track .mid file comprising a drums stem, bass stem and piano stem.",
                                        usage="<out_dir> <midifiles> -n <merged_filename>")
    
    argparser.add_argument("out_dir", nargs=1, help="Provide an output directory for the merged MIDI file")
    argparser.add_argument("midifiles", nargs="*", help="Provide the paths to each of the drums.mid, bass.mid and piano.mid files. Ensure that each .mid file has one (and only one of) 'drums', 'bass' or 'piano' in its filename.")
    argparser.add_argument("-n", nargs=1, help="Use option if you wish to provide a merged_filename (otherwise default merged .mid filename will be 'merged.mid').")
    
    args = argparser.parse_args()

    if len(args.midifiles) > 3:
        raise TypeError("Please provide a maximum of 3 arguments")
    
    if args.n:
        merge_midi(midifiles=args.midifiles, out_dir=args.out_dir[0], merged_filename=args.n[0])
    else:
        merge_midi(midifiles=args.midifiles, out_dir=args.out_dir[0])
        
    '''
    # split a track to test merge_midi on it
    track = parser.MidiFile("test/song0.mid")
    
    for ins in track.instruments:
        new_track = parser.MidiFile(ticks_per_beat=track.ticks_per_beat)
        new_track.instruments.append(ins)
        pprint(ins.notes)
        print("==========================================")
        pprint(new_track.instruments[0].notes)
        print("==========================================")
        new_track.dump(f"test/merge_midi/{ins.name}.mid")
    '''

if __name__ == "__main__":

    main()

