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

class MidiTokenizer(MidiTokenizerBase):
    """
    bar-ins-startpos-notedur-pitch
    bars go from 0 - 128
    """
    def __init__(self):
        super().__init__()
        with open("encoding_nopool.json") as file:
            self.vocab = json.load(file)

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
        res["ids"].append(self.vocab["BOS_None"])
        res["events"].append("<<BOS>>")

        bar_count = 0

        for note in notes_all:

            # BAR
            if note.start // (tpb * 4) >= (bar_count + 1):
                bar_count += 1
            res["ids"].append(self.vocab[f"bar{bar_count}"])
            res["events"].append(f"BAR{bar_count}")

            # INSTRUMENT
            res["ids"].append(self.vocab[note.instrument])
            res["events"].append(note.instrument)

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            res["ids"].append(self.vocab[f"start_pos{start_pos}"])
            res["events"].append(f"start_pos{start_pos}")

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            res["ids"].append(self.vocab[f"note_dur{note_dur}"])
            res["events"].append(f"note_dur{note_dur}")

            # PITCH
            res["ids"].append(self.vocab[f"pitch{note.pitch}"])
            res["events"].append(f"pitch{note.pitch}")

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

            if dict_rev[str(token)] in ['PAD_None','BOS_None','EOS_None','Mask_None']:
                counter += 1
                continue

            # create Note object
            if 'bar' in dict_rev[str(token)] and 'pitch' in dict_rev[str(tokens_list[counter+4])]:
                bar_num = int(dict_rev[str(token)][3:])
                start = int(dict_rev[str(tokens_list[counter+2])][9:]) * (1/8) * tpb + (bar_num * tpb * 4)
                end = start + (int(dict_rev[str(tokens_list[counter+3])][8:]) * (1/8) * tpb)
                pitch = int(dict_rev[str(tokens_list[counter+4])][5:])
                note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

                inst_idx = inst[dict_rev[str(tokens_list[counter+1])]]
                res.instruments[inst_idx].notes.append(note)

                counter += 5

            else:
                print("Counter: ", counter, " Tokens List: ", tokens_list[counter])
                error_rate += 1
                counter += 1

        print("Notes not transcribed: ", error_rate)
        return res
    

class MidiTokenizer2(MidiTokenizerBase):
    """
    ins-startpos-notedur-pitch
    """
    def __init__(self):
        super().__init__()
        with open("encoding_nopool2.json") as file:
            self.vocab = json.load(file)


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
        res["ids"].append(self.vocab["BOS_None"])
        res["events"].append("<<BOS>>")

        # Add first BAR
        res["ids"].append(self.vocab["BAR"])
        res["events"].append("|==BAR==|")

        bar_count = 1

        for note in notes_all:

            # Add BAR if required
            if note.start // (tpb * 4) >= bar_count:
                res["ids"].append(self.vocab["BAR"])
                res["events"].append("|==BAR==|")
                bar_count += 1

            # INSTRUMENT
            res["ids"].append(self.vocab[note.instrument])
            res["events"].append(note.instrument)

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            res["ids"].append(self.vocab[f"start_pos{start_pos}"])
            res["events"].append(f"start_pos{start_pos}")

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            res["ids"].append(self.vocab[f"note_dur{note_dur}"])
            res["events"].append(f"note_dur{note_dur}")

            # PITCH
            res["ids"].append(self.vocab[f"pitch{note.pitch}"])
            res["events"].append(f"pitch{note.pitch}")

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
        bar = -1
        while counter < len(tokens_list) - 3: # change this if changing tokenizer format
            token = tokens_list[counter]

            if dict_rev[str(token)] in ['PAD_None','BOS_None','EOS_None','Mask_None']:
                counter += 1
                continue
            
            # update bar count
            if dict_rev[str(token)] == 'BAR':
                bar += 1
                counter +=1
                continue

            # create Note object
            if dict_rev[str(token)] in inst and 'pitch' in dict_rev[str(tokens_list[counter+3])]:
                start = int(dict_rev[str(tokens_list[counter+1])][9:]) * (1/8) * tpb + (bar * tpb * 4)
                end = start + (int(dict_rev[str(tokens_list[counter+2])][8:]) * (1/8) * tpb)
                pitch = int(dict_rev[str(tokens_list[counter+3])][5:])
                note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

                inst_idx = inst[dict_rev[str(token)]]
                res.instruments[inst_idx].notes.append(note)

                counter += 4

            else:
                print("Counter: ", counter, " Tokens List: ", tokens_list[counter])
                error_rate += 1
                counter += 1

        print("Notes not transcribed: ", error_rate)
        return res


class MidiTokenizer3(MidiTokenizerBase):
    """
    bar-startpos-notedur-pitch-ins
    bars go from 0 - 128
    """
    def __init__(self):
        super().__init__()
        with open("encoding_nopool.json") as file:
            self.vocab = json.load(file)


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
        res["ids"].append(self.vocab["BOS_None"])
        res["events"].append("<<BOS>>")

        bar_count = 0

        for note in notes_all:

            # BAR
            if note.start // (tpb * 4) >= (bar_count + 1):
                bar_count += 1
            res["ids"].append(self.vocab[f"bar{bar_count}"])
            res["events"].append(f"BAR{bar_count}")

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            res["ids"].append(self.vocab[f"start_pos{start_pos}"])
            res["events"].append(f"start_pos{start_pos}")

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            res["ids"].append(self.vocab[f"note_dur{note_dur}"])
            res["events"].append(f"note_dur{note_dur}")

            # PITCH
            res["ids"].append(self.vocab[f"pitch{note.pitch}"])
            res["events"].append(f"pitch{note.pitch}")

            # INSTRUMENT
            res["ids"].append(self.vocab[note.instrument])
            res["events"].append(note.instrument)

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

            if dict_rev[str(token)] in ['PAD_None','BOS_None','EOS_None','Mask_None']:
                counter += 1
                continue

            # create Note object
            if 'bar' in dict_rev[str(token)] and dict_rev[str(tokens_list[counter+4])] in ['drums', 'bass', 'piano']:
                bar_num = int(dict_rev[str(token)][3:])
                start = int(dict_rev[str(tokens_list[counter+1])][9:]) * (1/8) * tpb + (bar_num * tpb * 4)
                end = start + (int(dict_rev[str(tokens_list[counter+2])][8:]) * (1/8) * tpb)
                pitch = int(dict_rev[str(tokens_list[counter+3])][5:])
                note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

                inst_idx = inst[dict_rev[str(tokens_list[counter+4])]]
                res.instruments[inst_idx].notes.append(note)

                counter += 5

            else:
                print("Counter: ", counter, " Tokens List: ", tokens_list[counter])
                error_rate += 1
                counter += 1

        print("Notes not transcribed: ", error_rate)
        return res


class MidiTokenizer4(MidiTokenizerBase):
    """
    startpos-notedur-pitch-ins
    """
    def __init__(self):
        super().__init__()
        with open("encoding_nopool2.json") as file:
            self.vocab = json.load(file)


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
        res["ids"].append(self.vocab["BOS_None"])
        res["events"].append("<<BOS>>")

        # Add first BAR
        res["ids"].append(self.vocab["BAR"])
        res["events"].append("|==BAR==|")

        bar_count = 1

        for note in notes_all:

            # Add BAR if required
            if note.start // (tpb * 4) >= bar_count:
                res["ids"].append(self.vocab["BAR"])
                res["events"].append("|==BAR==|")
                bar_count += 1

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            res["ids"].append(self.vocab[f"start_pos{start_pos}"])
            res["events"].append(f"start_pos{start_pos}")

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            res["ids"].append(self.vocab[f"note_dur{note_dur}"])
            res["events"].append(f"note_dur{note_dur}")

            # PITCH
            res["ids"].append(self.vocab[f"pitch{note.pitch}"])
            res["events"].append(f"pitch{note.pitch}")

            # INSTRUMENT
            res["ids"].append(self.vocab[note.instrument])
            res["events"].append(note.instrument)

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
        bar = -1
        while counter < len(tokens_list) - 3: # change this if changing tokenizer format
            token = tokens_list[counter]

            if dict_rev[str(token)] in ['PAD_None','BOS_None','EOS_None','Mask_None']:
                counter += 1
                continue
            
            # update bar count
            if dict_rev[str(token)] == 'BAR':
                bar += 1
                counter +=1
                continue

            # create Note object
            if 'start_pos' in dict_rev[str(token)] and dict_rev[str(tokens_list[counter+3])] in ['drums', 'bass', 'piano']:
                start = int(dict_rev[str(tokens_list[counter])][9:]) * (1/8) * tpb + (bar * tpb * 4)
                end = start + (int(dict_rev[str(tokens_list[counter+1])][8:]) * (1/8) * tpb)
                pitch = int(dict_rev[str(tokens_list[counter+2])][5:])
                note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

                inst_idx = inst[dict_rev[str(tokens_list[counter+3])]]
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
    Uses embedding pooling
    Token format (start_pos, note_dur, pitch, instrument)
    """
    def __init__(self):
        super().__init__()
        with open("encoding_pooled.json") as file:
            self.vocab = json.load(file)

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
            print("Skipped: ", filepath, " - failed to parse mid file")
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
        res["ids"].append((self.vocab["start_pos"]["BOS_None"],
                        self.vocab["note_dur"]["BOS_None"],
                        self.vocab["pitch"]["BOS_None"],
                        self.vocab["instrument"]["BOS_None"]))
        res["events"].append("<<BOS>>")

        # Add first BAR
        res["ids"].append((self.vocab["start_pos"]["BAR"],
                        self.vocab["note_dur"]["BAR"],
                        self.vocab["pitch"]["BAR"],
                        self.vocab["instrument"]["BAR"]))
        res["events"].append("|==BAR==|")

        bar_count = 1

        for note in notes_all:

            # Add BAR if required
            if note.start // (tpb * 4) >= bar_count:
                res["ids"].append((self.vocab["start_pos"]["BAR"],
                            self.vocab["note_dur"]["BAR"],
                            self.vocab["pitch"]["BAR"],
                            self.vocab["instrument"]["BAR"]))
                res["events"].append("|==BAR==|")
                bar_count += 1

            # START_POS
            start_pos = int((note.start % (tpb * 4)) / (tpb / 8))
            start_pos_temp = start_pos
            start_pos = self.vocab["start_pos"][f"start_pos{start_pos}"]

            # NOTE_DURATION
            note_dur = int(note.get_duration() / tpb / (1/8))
            note_dur = self.vocab["note_dur"][f"note_dur{note_dur}"]

            # PITCH
            pitch = self.vocab["pitch"][f"pitch{note.pitch}"]

            # INSTRUMENT
            instrument = self.vocab["instrument"][note.instrument]

            # COMBINE
            res["ids"].append((start_pos,
                            note_dur,
                            pitch,
                            instrument))
            res["events"].append((f"start_pos{start_pos_temp}",
                                f"note_dur{note_dur}",
                                f"pitch{note.pitch}",
                                note.instrument))

        # EOS
        res["ids"].append((self.vocab["start_pos"]["EOS_None"],
                        self.vocab["note_dur"]["EOS_None"],
                        self.vocab["pitch"]["EOS_None"],
                        self.vocab["instrument"]["EOS_None"]))
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
            tokens_list: must be a list or tensor of size T x 4
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

            # BAR
            # if one token appears in the above, they all should, otherwise skip
            # either way, escape loop
            if tokens[0] == 4 or tokens[1] == 4 or tokens[2] == 4 or tokens[3] == 4:
                if sum([token == 4 for token in tokens]) != 4:
                    print("Counter: ", counter, " Tokens List: ", tokens)
                    error_rate += 1
                else:
                    bar += 1

                counter += 1
                continue

            # create Note object
            start = int(dict_rev["start_pos"][f"{tokens[0]}"][9:]) * (1/8) * tpb + (bar * tpb * 4)
            end = start + (int(dict_rev["note_dur"][f"{tokens[1]}"][8:]) * (1/8) * tpb)
            pitch = int(dict_rev["pitch"][f"{tokens[2]}"][5:])
            note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

            # assign to the right instrument
            inst = {"drums": 0, "bass": 1, "piano": 2}
            inst_idx = inst[dict_rev["instrument"][f"{tokens[3]}"]]
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
            start_pos = dict_rev['start_pos'][f'{tokens[0]}']
            note_dur = dict_rev['note_dur'][f'{tokens[1]}']
            pitch = dict_rev['pitch'][f'{tokens[2]}']
            instrument = dict_rev['instrument'][f'{tokens[3]}']

            res.append((start_pos, note_dur, pitch, instrument))

        return res

class MidiTokenizerPooled2(MidiTokenizerBase):
    """
    bar-startpos-notedur-pitch-ins
    """
    def __init__(self):
        with open("encoding_pooled_withbartokens.json") as file:
            self.vocab = json.load(file)

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
                        self.vocab["pitch"]["BOS_None"],
                        self.vocab["instrument"]["BOS_None"]))
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

            # PITCH
            pitch = self.vocab["pitch"][f"pitch{note.pitch}"]

            # INSTRUMENT
            instrument = self.vocab["instrument"][note.instrument]

            # COMBINE
            res["ids"].append((bar,
                            start_pos,
                            note_dur,
                            pitch,
                            instrument))
            res["events"].append((f"bar{bar_temp}",
                                f"start_pos{start_pos_temp}",
                                f"note_dur{note_dur}",
                                f"pitch{note.pitch}",
                                note.instrument))

        # EOS
        res["ids"].append((self.vocab["bar"]["EOS_None"],
                        self.vocab["start_pos"]["EOS_None"],
                        self.vocab["note_dur"]["EOS_None"],
                        self.vocab["pitch"]["EOS_None"],
                        self.vocab["instrument"]["EOS_None"]))
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
                if sum([token in [0, 1, 2, 3] for token in tokens]) != 5:
                    print("Counter: ", counter, " Tokens List: ", tokens)
                    error_rate += 1
                counter += 1
                continue

            # create Note object
            bar = int(dict_rev["bar"][f"{tokens[0]}"][3:])
            start = int(dict_rev["start_pos"][f"{tokens[1]}"][9:]) * (1/8) * tpb + (bar * tpb * 4)
            end = start + (int(dict_rev["note_dur"][f"{tokens[2]}"][8:]) * (1/8) * tpb)
            pitch = int(dict_rev["pitch"][f"{tokens[3]}"][5:])
            note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

            # assign to the right instrument
            inst = {"drums": 0, "bass": 1, "piano": 2}
            inst_idx = inst[dict_rev["instrument"][f"{tokens[4]}"]]
            res.instruments[inst_idx].notes.append(note)

            counter += 1

        print("Notes not transcribed: ", error_rate)
        return res

    def tokens_to_events(self, tokens_list: Union[list[Union[tuple[int],list[int]]],torch.Tensor]) -> list[tuple[str]]:
        """
        takes list of token ids and returns events
        inputs should be T x 5 - can be list of lists, list of tuples or tensor of size T x 5
        """
        # build reverse vocab dict
        dict_rev = {token_fam: {str(v): k for k, v in self.vocab[token_fam].items()} for token_fam in list(self.vocab.keys())[1:]}

        res = []
        for tokens in tokens_list:
            bar = dict_rev['bar'][f'{tokens[0]}']
            start_pos = dict_rev['start_pos'][f'{tokens[1]}']
            note_dur = dict_rev['note_dur'][f'{tokens[2]}']
            pitch = dict_rev['pitch'][f'{tokens[3]}']
            instrument = dict_rev['instrument'][f'{tokens[4]}']

            res.append((bar, start_pos, note_dur, pitch, instrument))

        return res

class MidiTokenizerPooled3(MidiTokenizerBase):
    """
    bar-startpos-notedur-pitchins (pitch and ins are concat)
    """
    def __init__(self):
        with open("encoding_pooled_pitch_ins.json") as file:
            self.vocab = json.load(file)

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
            pitch = int(dict_rev["pitch_instrument"][f"{tokens[3]}"][5:7])
            note = ct.Note(start=int(start), end=int(end), pitch=pitch, velocity=100)

            # assign to the right instrument
            inst = {"drums": 0, "bass": 1, "piano": 2}
            inst_idx = inst[dict_rev["pitch_instrument"][f"{tokens[3]}"][8:]]
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
    
def tokenize_dataset(tokenizer: MidiTokenizerBase, tokens_folder: Union[str,Path]):
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
    for i in Path('dataset').glob('*[0-9].mid'):
            tokenizer(i, tokens_folder.joinpath(f'{i.stem}.json'))

    # tokenize LMD
    lmd_path = Path('lmd_cleaned2')
    for i in tqdm(lmd_path.glob('*.mid')):
            try:
                tokenizer(i, tokens_folder.joinpath(f'{i.stem}.json'))
            except Exception as e:
                print(f"Skipped: {i.stem} due to {e}")
                continue
    
    print(f"Total Songs: {list(tokens_folder.glob('*.json')).__len__()}")

def test_tokenizer(tokenizer: MidiTokenizerBase):
    tokenizer = tokenizer()
    song0_tokenizer_tokens = tokenizer('song0.mid')
    print(song0_tokenizer_tokens['ids'])
    print(tokenizer.tokens_to_events(song0_tokenizer_tokens['ids']))
    tokenizer.create_midi_from_tokens(song0_tokenizer_tokens['ids']).dump("song0_regen_tok1.mid")

if __name__ == "__main__":

    #test_tokenizer(MidiTokenizerPooled3)

    tokenize_dataset(MidiTokenizerPooled3, "tokens_pooled_pitch_ins")

    # test MidiTokenizerPooled
    '''
    tokenizer = MidiTokenizerPooled()
    song0_tokenizer_tokens = tokenizer('song0.mid', Path('song0.json'))['ids']
    tokenizer.create_midi_from_tokens(song0_tokenizer_tokens).dump("song0_regen_tok1.mid")
    '''

    # test
    '''
    tokenizer = MidiTokenizer()
    tokenizer2 = MidiTokenizer2()
    tokenizer3 = MidiTokenizer3()
    tokenizer4 = MidiTokenizer4()

    song0_tokenizer_tokens = tokenizer('song0.mid')['ids']
    song0_tokenizer2_tokens = tokenizer2('song0.mid')['ids']
    song0_tokenizer3_tokens = tokenizer3('song0.mid')['ids']
    song0_tokenizer4_tokens = tokenizer4('song0.mid')['ids']

    tokenizer.create_midi_from_tokens(song0_tokenizer_tokens).dump("song0_regen_tok1.mid")
    tokenizer2.create_midi_from_tokens(song0_tokenizer2_tokens).dump("song0_regen_tok2.mid")
    tokenizer3.create_midi_from_tokens(song0_tokenizer3_tokens).dump("song0_regen_tok3.mid")
    tokenizer4.create_midi_from_tokens(song0_tokenizer4_tokens).dump("song0_regen_tok4.mid")
    '''