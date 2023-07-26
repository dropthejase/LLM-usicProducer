import json
from pathlib import Path
from tqdm import tqdm
from typing import Union

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
    
if __name__ == "__main__":

    '''
    tokenizer4 = MidiTokenizer4()
    tokens_path = Path('tokens4')

    # tokenize my own dataset
    for i in Path('dataset').glob('*[0-9].mid'):
            tokenizer4(i, tokens_path.joinpath(f'{i.stem}.json'))

    # tokenize LMD
    lmd_path = Path('lmd_cleaned2')
    for i in tqdm(lmd_path.glob('*.mid')):
            try:
                    tokenizer4(i, tokens_path.joinpath(f'{i.stem}.json'))
            except Exception as e:
                    print(f"Skipped: {i.stem} due to {e}")
                    continue

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