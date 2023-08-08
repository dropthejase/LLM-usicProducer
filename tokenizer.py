import argparse
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
from typing import Union

import torch

from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct


class Note(ct.Note):
    def __init__(self, note: ct.Note, instrument: str):
        super().__init__(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity)

        self.instrument = instrument

    def __repr__(self):
        return 'Note(instrument={}, start={:d}, end={:d}, pitch={}, velocity={})'.format(
                self.instrument, self.start, self.end, self.pitch, self.velocity)

class MidiTokenizerBase:
    """base class for tokenizers"""
    def __init__(self):
        pass
    
    def create_vocab(self, vocab: Union[str,Path,dict]) -> dict:
        if isinstance(vocab, dict):
            return dict
        else:
            if isinstance(vocab, str):
                vocab = Path(vocab)
            if not vocab.exists() or vocab.suffix not in ['.json']:
                raise FileNotFoundError('Please provide a valid filepath to the json file containing the vocab list')
            with open(vocab) as file:
                return json.load(file)

    @staticmethod
    def quantize(midifile: parser.MidiFile) -> parser.MidiFile:
        """quantizes MidiFile to 1/32 and returns a new one"""

        step = int(midifile.ticks_per_beat / 8)

        for ins in midifile.instruments:
            for note in ins.notes:
                dur = note.get_duration()

                # quantize note duration
                # min 1/32
                if dur < step:
                    dur = step
                # max 4 bars
                if dur > (step * 32 * 4):
                    dur = step * 32 * 4
                # stretch down otherwise stretch up
                if dur % step != 0:
                    if (dur % step) <= (step / 2):
                        dur = (dur // step) * step
                    else:
                        dur = ((dur // step) + 1) * step
                note.end = note.start + dur

                # shift position to nearest 1/32
                if note.start % step != 0:
                    shift_back = step * (note.start // step)
                    shift_forward = step * ((note.start // step) + 1)

                    if (note.start - shift_back) <= (shift_forward - note.start):
                        note.start = shift_back
                    else:
                        note.start = shift_forward

                    note.end = note.start + dur

        return midifile

    def tokens_to_events(self, tokens_list: list[int]) -> list[str]:
        """takes list of token ids and returns events"""
        dict_rev = {str(v): k for k,v in self.vocab.items()}

        res = []
        for tokens in tokens_list:
            res.append(dict_rev[f'{tokens}'])

        return res


class MidiTokenizerNoPool(MidiTokenizerBase):
    """
    bar-startpos-notedur-pitch_ins
    bars go from 0 - 255
    """
    def __init__(self, vocab: Union[str,Path,dict]="encoding_nopool_pitch_ins.json"):
        """
        vocab: can be a dict variable, or a str or Path to the .json file with the vocab list
        """
        super().__init__()
        self.vocab = self.create_vocab(vocab)

    def __call__(self, midifile: Union[str,Path,parser.MidiFile], out_dir: Union[str,Path]=None) -> dict:
        """
        tokenizes either a MidiFile object or a path to the .mid file
        returns dict comprising {"ids": list of ids, "events": list of events}
        """

        if not isinstance(midifile, parser.MidiFile):

            # check file exists
            if not isinstance(midifile, Path):
                midifile = Path(midifile)
            if not midifile.exists() or midifile.suffix not in ['.mid']:
                raise FileNotFoundError('Please provide filepath to the .mid file')

            # create Midi parser
            try:
                midifile = parser.MidiFile(midifile)
            except:
                print("Skipped: ", midifile)
                return

        if midifile.ticks_per_beat % 8 != 0:
            print(f"Skipped (TPB = {midifile.ticks_per_beat}): ", midifile)
            return

        # quantize
        midifile = self.quantize(midifile)

        # parse MidiFile, create Note objects, and collate
        notes_all = []
        for ins in midifile.instruments:
            for note in ins.notes:
                note = Note(note, ins.name)
                notes_all.append(note)

        # sort all notes (in place) by start times
        # this allows us to put BAR tokens properly
        notes_all.sort(key=lambda x: x.start)

        # tokenize
        res = {"ids": [], "events": []}
        tpb = midifile.ticks_per_beat

        # BOS
        res["ids"].append(self.vocab["BOS_None"])
        res["events"].append("<<BOS>>")

        bar_count = 0

        for note in notes_all:

            # BAR
            if note.start // (tpb * 4) >= (bar_count + 1):
                bar_count += 1
            res["ids"].append(self.vocab[f"bar{bar_count}"])
            res["events"].append(f"bar{bar_count}")

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            res["ids"].append(self.vocab[f"start_pos{start_pos}"])
            res["events"].append(f"start_pos{start_pos}")

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            res["ids"].append(self.vocab[f"note_dur{note_dur}"])
            res["events"].append(f"note_dur{note_dur}")

            # PITCH_INSTRUMENT
            res["ids"].append(self.vocab[f"pitch{note.pitch}_{note.instrument}"])
            res["events"].append(f"pitch{note.pitch}_{note.instrument}")

        # EOS
        res["ids"].append(self.vocab["EOS_None"])
        res["events"].append("<<EOS>>")

        # Save as json
        if out_dir:
            with open(out_dir, 'w') as output:
                json.dump(res, output)
        return res

    def create_midi_from_tokens(self, tokens_list: list[int], tpb: int=96) -> parser.MidiFile:
        """creates MidiFile from list of token ids"""

        # set up final MidiFile
        res = parser.MidiFile(ticks_per_beat=tpb)
        res.instruments = [
                ct.Instrument(program=0, is_drum=True, name="drums"),
                ct.Instrument(program=38, is_drum=False, name="bass"),
                ct.Instrument(program=0, is_drum=False, name="piano")
        ]
        res.time_signature_changes = [ct.TimeSignature(numerator=4, denominator=4, time=0)]

        # build reverse vocab dict
        dict_rev = {str(v): k for k,v in self.vocab.items()}

        # create dict for instruments mapping instrument to index in res.instruments
        inst = {"drums": 0, "bass": 1, "piano": 2}

        counter = 0
        error_rate = 0
        while counter < len(tokens_list) - 4: # change this if changing tokenizer format
            token = tokens_list[counter]

            if dict_rev[f'{token}'] in ['PAD_None','BOS_None','EOS_None','Mask_None']:
                counter += 1
                continue

            # create Note object
            if 'bar' in dict_rev[f'{token}'] and \
            'start_pos' in dict_rev[f'{tokens_list[counter+1]}'] and \
            'note_dur' in dict_rev[f'{tokens_list[counter+2]}'] and \
            tokens_list[counter+3] >= 292:
                
                bar_num = int(dict_rev[f'{token}'][3:])
                start = int(dict_rev[f'{tokens_list[counter+1]}'][9:]) * (1/8) * tpb + (bar_num * tpb * 4)
                end = start + (int(dict_rev[f'{tokens_list[counter+2]}'][8:]) * (1/8) * tpb)

                pitch_instrument = dict_rev[f'{tokens_list[counter+3]}'][5:].split("_")
                pitch, instrument = pitch_instrument[0], pitch_instrument[1]
                pitch = int(pitch)

                note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

                inst_idx = inst[instrument]
                res.instruments[inst_idx].notes.append(note)

                counter += 4

            else:
                print("Counter: ", counter, " Tokens List: ", tokens_list[counter])
                error_rate += 1
                counter += 1

        print("Notes not transcribed: ", error_rate)
        return res


class MidiTokenizerPooled(MidiTokenizerBase):
    """
    bar-startpos-notedur-pitch_ins (pitch and ins are concat)
    """
    def __init__(self, vocab: Union[str,Path,dict]="encoding_pooled_pitch_ins.json"):
        """
        vocab: can be a dict variable, or a str or Path to the .json file with the vocab list
        """
        self.vocab = self.create_vocab(vocab)

    def __call__(self, filepath: Union[str,Path], out_dir: Union[str,Path]=None) -> dict:
        """
        tokenizes midifile
        return dict comprising {"ids": list of ids, "events": list of events}
        """

        # check file exists
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if not filepath.exists() or filepath.suffix not in ['.mid']:
            raise FileNotFoundError('Please provide filepath to the .mid file')

        # create Midi parser
        try:
            midifile = parser.MidiFile(filepath)
        except:
            print("Skipped: ", filepath)
            return

        if midifile.ticks_per_beat % 8 != 0:
            print(f"Skipped (TPB = {midifile.ticks_per_beat}): ", filepath)
            return

        # quantize
        midifile = self.quantize(midifile)

        # parse MidiFile, create Note objects, and collate
        notes_all = []
        for ins in midifile.instruments:
            for note in ins.notes:
                note = Note(note, ins.name)
                notes_all.append(note)

        # sort all notes (in place) by start times
        # this allows us to put BAR tokens properly
        notes_all.sort(key=lambda x: x.start)

        # tokenize
        res = {"ids": [], "events": []}
        tpb = midifile.ticks_per_beat

        # BOS
        res["ids"].append((self.vocab["bar"]["BOS_None"],
                        self.vocab["start_pos"]["BOS_None"],
                        self.vocab["note_dur"]["BOS_None"],
                        self.vocab["pitch_instrument"]["BOS_None"]))
        res["events"].append("<<BOS>>")

        for note in notes_all:

            # BAR
            bar = note.start // (tpb * 4)
            bar_temp = bar
            bar = self.vocab["bar"][f"bar{bar}"]

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            start_pos_temp = start_pos
            start_pos = self.vocab["start_pos"][f"start_pos{start_pos}"]

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            note_dur = self.vocab["note_dur"][f"note_dur{note_dur}"]

            # PITCH-INSTRUMENT
            pitch_instrument = self.vocab["pitch_instrument"][f"pitch{note.pitch}_{note.instrument}"]

            # COMBINE
            res["ids"].append((bar,
                            start_pos,
                            note_dur,
                            pitch_instrument))
            res["events"].append((f"bar{bar_temp}",
                                f"start_pos{start_pos_temp}",
                                f"note_dur{note_dur}",
                                f"pitch{note.pitch}_{note.instrument}"))

        # EOS
        res["ids"].append((self.vocab["bar"]["EOS_None"],
                        self.vocab["start_pos"]["EOS_None"],
                        self.vocab["note_dur"]["EOS_None"],
                        self.vocab["pitch_instrument"]["EOS_None"]))
        res["events"].append("<<EOS>>")

        # Save as json
        if out_dir:
            with open(out_dir, 'w') as output:
                json.dump(res, output)

        return res

    def create_midi_from_tokens(self, tokens_list: Union[list[int],torch.Tensor], tpb: int=96) -> parser.MidiFile:
        """
        creates MidiFile from list of token ids
        inputs:
            tokens_list: must be a list or tensor of size T x 5
        """

        # set up final MidiFile
        res = parser.MidiFile(ticks_per_beat=tpb)
        res.instruments = [
            ct.Instrument(program=0, is_drum=True, name="drums"),
            ct.Instrument(program=38, is_drum=False, name="bass"),
            ct.Instrument(program=0, is_drum=False, name="piano")
        ]
        res.time_signature_changes = [ct.TimeSignature(numerator=4, denominator=4, time=0)]

        # build reverse vocab dict (ignores n_tokens key)
        dict_rev = {token_fam: {str(v): k for k, v in self.vocab[token_fam].items()} for token_fam in list(self.vocab.keys())[1:]}

        counter = 0
        bar = -1
        error_rate = 0
        while counter < len(tokens_list):
            tokens = tokens_list[counter] # (start_pos, note_dur, pitch, instrument)

            # ['PAD_None','BOS_None','EOS_None','Mask_None']
            # if one token appears in the above, they all should, otherwise skip
            # either way, escape loop
            if tokens[0] in [0, 1, 2, 3] or tokens[1] in [0, 1, 2, 3] or tokens[2] in [0, 1, 2, 3] or tokens[3] in [0, 1, 2, 3]:
                if sum([token in [0, 1, 2, 3] for token in tokens]) != 4:
                    print("Counter: ", counter, " Tokens List: ", tokens)
                    error_rate += 1
                counter += 1
                continue

            # create Note object
            bar = int(dict_rev["bar"][f"{tokens[0]}"][3:])
            start = int(dict_rev["start_pos"][f"{tokens[1]}"][9:]) * (1/8) * tpb + (bar * tpb * 4)
            end = start + (int(dict_rev["note_dur"][f"{tokens[2]}"][8:]) * (1/8) * tpb)
            
            pitch_instrument = dict_rev["pitch_instrument"][f"{tokens[3]}"][5:].split("_")
            pitch, instrument = pitch_instrument[0], pitch_instrument[1]
            pitch = int(pitch)
            
            note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

            # assign to the right instrument
            inst = {"drums": 0, "bass": 1, "piano": 2}

            inst_idx = inst[instrument]
            res.instruments[inst_idx].notes.append(note)

            counter += 1

        print("Notes not transcribed: ", error_rate)
        return res

    def tokens_to_events(self, tokens_list: Union[list[Union[tuple[int],list[int]]],torch.Tensor]) -> list[tuple[str]]:
        """
        takes list of token ids and returns events
        inputs should be T x 4 - can be list of lists, list of tuples or tensor of size T x 4
        """
        # build reverse vocab dict
        dict_rev = {token_fam: {str(v): k for k, v in self.vocab[token_fam].items()} for token_fam in list(self.vocab.keys())[1:]}

        res = []
        for tokens in tokens_list:
            bar = dict_rev['bar'][f'{tokens[0]}']
            start_pos = dict_rev['start_pos'][f'{tokens[1]}']
            note_dur = dict_rev['note_dur'][f'{tokens[2]}']
            pitch_instrument = dict_rev['pitch_instrument'][f'{tokens[3]}']

            res.append((bar, start_pos, note_dur, pitch_instrument))

        return res
    

def tokenize_dataset(tokenizer: MidiTokenizerBase, tokens_folder: Union[str,Path]="tokens", midi_files: Union[str,Path,list[Union[str,Path]]]="lmd_cleaned2"):
    """
    Function to tokenize my dataset and LMD dataset
    Inputs
        tokens_folder: output directory for tokenized json files
    """
    if not isinstance(tokens_folder, Path):
        tokens_folder = Path(tokens_folder)
    if not tokens_folder.exists():
        tokens_folder.mkdir()
    
    tokenizer = tokenizer()
    
    # tokenize my own dataset
    #for i in Path('dataset').glob('*[0-9].mid'):
    #        tokenizer(i, tokens_folder.joinpath(f'{i.stem}.json'))

    # if single file provided, turn into a list
    if not isinstance(midi_files, list):
        if not isinstance(midi_files, Path):
            midi_files = Path(midi_files)
        if not midi_files.exists():
            raise FileNotFoundError("Please provide a valid path to the .mid file")
        midi_files = [midi_files]

    # create text file
    with open(f"{tokens_folder}/logs.txt", "a") as f:
        f.write("=================================================================")
        f.write("\n")
        f.write(f"{datetime.now()}")
        f.write("\n")
        f.write("=================================================================")
        f.write("\n")
        f.write("\n")

    # tokenize
    for file in tqdm(midi_files):
        
        # check file or folder exists
        if not isinstance(file, Path):
            file = Path(file)
        if not file.exists():
            print(f"Skipped: {file.stem} as file could not be located")
            continue
        
        # if list item is a single .mid
        if file.suffix == ".mid":
            try:
                tokenizer(file, tokens_folder.joinpath(f'{file.stem}.json'))
            except Exception as e:
                with open(f"{tokens_folder}/logs.txt", "a") as f:
                    f.write(f"\n Skipped: {file.stem} due to {e}")
                print(f"Skipped: {file.stem} due to {e}")
                continue
        
        # if list item is a folder
        else:
            for i in tqdm(file.glob('*.mid')):
                try:
                    tokenizer(i, tokens_folder.joinpath(f'{i.stem}.json'))
                except Exception as e:
                    with open(f"{tokens_folder}/logs.txt", "a") as f:
                        f.write(f"\n Skipped: {i.stem} due to {e}")
                    print(f"Skipped: {i.stem} due to {e}")
                    continue
    
    print(f"Total Songs: {list(tokens_folder.glob('*.json')).__len__()}")


def test_tokenizer(tokenizer: MidiTokenizerBase):
    tokenizer = tokenizer()
    song0_tokenizer_tokens = tokenizer('test/song0.mid')
    print(song0_tokenizer_tokens)
    print(song0_tokenizer_tokens['ids'])
    print(tokenizer.tokens_to_events(song0_tokenizer_tokens['ids']))
    tokenizer.create_midi_from_tokens(song0_tokenizer_tokens['ids']).dump("test/song0_regen_tok1.mid")


def main():

    argparser = argparse.ArgumentParser(prog="Tokenizer", description="To tokenize MIDI files")
    argparser.add_argument("-p", "--pooled", action='store_true', required=False, help="If used, will use MidiTokenizerPooled to tokenize dataset, otherwise will use MidiTokenizerNoPool")
    argparser.add_argument("-td", "--tokenize_dataset", action='store_true', required=False, help="If used, will run tokenize_dataset function")
    argparser.add_argument("tokens_folder", nargs="?", default="tokens", help="Provide an output directory for .json files containing tokenized dataset (default: a folder named 'tokens')")
    argparser.add_argument("midi_files", nargs="*", default="lmd_cleaned2", help="Provide .mid files or folders containing .mid files to tokenize when -td option used (default: 'lmd_cleaned2')")
    
    ## TO DO - add dataset output_dir

    args = argparser.parse_args()
    print(args)

    if args.pooled:
        tokenizer = MidiTokenizerPooled
    else:
        tokenizer = MidiTokenizerNoPool

    if not args.tokenize_dataset:
        test_tokenizer(tokenizer)
    
    else:
        if not args.midi_files:
            tokenize_dataset(tokenizer, tokens_folder=args.tokens_folder)
        else:
            tokenize_dataset(tokenizer, tokens_folder=args.tokens_folder, midi_files=args.midi_files)


if __name__ == "__main__":
    
    main()



