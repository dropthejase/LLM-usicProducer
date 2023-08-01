import json
from pathlib import Path
from pprint import pprint
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import MidiTokenizerPooled, MidiTokenizerPooled2, MidiTokenizerPooled3
from musictransformer import MusicTransformer2

def generate_sample(json_file: Union[str,Path],
                    prompt_idx: int=500,
                    print_new_events: bool=False,
                    out_dir: Union[str,Path]=None,
                    save_prompt_separately: bool=True,
                    **kwargs) -> list[int]:
    
    
    with open(json_file) as jsonfile:
        testprompt = json.load(jsonfile)['ids']
    print("Length of original JSON file: ", len(testprompt))

    if prompt_idx > len(testprompt):
        prompt_idx = len(testprompt)
    prompt = torch.LongTensor(testprompt[:prompt_idx]).view(1, -1, 5)

    print("Prompt Size: ", prompt.size())
    print("Prompt from idx: ", prompt_idx)

    gen = model.generate(prompt, **kwargs)

    gen = gen.reshape(-1, 5).tolist()

    if print_new_events:
        print("===========================NEW EVENTS================================")
        pprint(tokenizer.tokens_to_events(gen)[prompt_idx:], compact=True)

    if not out_dir:
        out_dir = 'gen.mid'
    tokenizer.create_midi_from_tokens(gen).dump(out_dir)

    if save_prompt_separately:
        tokenizer.create_midi_from_tokens(testprompt[:prompt_idx]).dump(f"{out_dir[:-4]}-prompt.mid")

    return gen

if __name__ == "__main__":

    device = torch.device("cpu")

    # load model
    tokenizer = MidiTokenizerPooled2()

    model = torch.load("musictransformer/musictransformer-full-3.pth")
    model.to(device)

 
    genconfig = {
        "temperature": [1.0, 1.0, 1.0, 1.0],
        "num_bars": 8,
        "max_steps": 128,
        "sampling_fn": "top_k",
        "threshold": [0.85, 0.85, 0.85, 0.85],
        "bar_token": 4
    }

    # note model misbehaves if prompt_idx is bigger than length of JSON as the last prompt is EOS. Do we need to pad?
    generate_sample("tokens_pooled_withbars/0.json", prompt_idx=256, print_new_events=True, out_dir="musictransformer/gen-0.mid", save_prompt_separately=True, **genconfig)
    generate_sample("tokens_pooled_withbars/16.json", prompt_idx=256, print_new_events=True, out_dir="musictransformer/gen-16.mid", save_prompt_separately=True, **genconfig)
    
    for i in range(3):
        generate_sample("noprompt_withbartokens.json", prompt_idx=2, print_new_events=True, out_dir=f"musictransformer/gen-noprompt{i}.mid", save_prompt_separately=False, **genconfig)
