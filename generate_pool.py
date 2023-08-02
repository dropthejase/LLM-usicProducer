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
                    num_token_families: int=4,
                    **kwargs) -> list[int]:
    
    
    with open(json_file) as jsonfile:
        testprompt = json.load(jsonfile)['ids']
    print("Length of original JSON file: ", len(testprompt))

    if prompt_idx > len(testprompt):
        prompt_idx = len(testprompt)
    prompt = torch.LongTensor(testprompt[:prompt_idx]).view(1, -1, num_token_families)

    print("Prompt Size: ", prompt.size())
    print("Prompt from idx: ", prompt_idx)

    gen = model.generate(prompt, **kwargs)

    gen = gen.reshape(-1, num_token_families).tolist()

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
    tokenizer = MidiTokenizerPooled3()

    model = torch.load("musictransformer/musictransformer-full-15.pth")
    model.to(device)

    #[0.6, 0.8, 0.8, 0.5 or 0.6] or [0.6, 0.7, 0.7, 0.8] - best so far for 512 for musictransformer-full-7
    # higher temperatures at [0.7, 0.9, 0.9, 0.9] seem to work better for a more trained model like musictransformer-full-15
    genconfig = {
        "temperature": [0.7, 0.9, 0.9, 0.9], #[0.6, 0.8, 0.8, 0.5 or 0.6] or [0.6, 0.7, 0.7, 0.8] - best so far for 512 for musictransformer-full-7
        "num_bars": 32,
        "max_steps": 1024,
        "sampling_fn": "top_k",
        "threshold": [0.9, 0.9, 0.9, 0.9],
        "bar_token": 4
    }

    # long prompts seem to be better, as well as lower temperatures (especially for bar token)
    
    #generate_sample("tokens_pooled_pitch_ins/0.json", prompt_idx=1024, print_new_events=True, num_token_families=4, out_dir="musictransformer/gen-0.mid", save_prompt_separately=True, **genconfig)
    #generate_sample("tokens_pooled_pitch_ins/16.json", prompt_idx=1024, print_new_events=True, num_token_families=4, out_dir="musictransformer/gen-16.mid", save_prompt_separately=True, **genconfig)
    #generate_sample("tokens_pooled_pitch_ins/20.json", prompt_idx=1024, print_new_events=True, num_token_families=4, out_dir="musictransformer/gen-20.mid", save_prompt_separately=True, **genconfig)
    #generate_sample("tokens_pooled_pitch_ins/24.json", prompt_idx=2056, print_new_events=True, num_token_families=4, out_dir="musictransformer/gen-24.mid", save_prompt_separately=True, **genconfig)
    
    for i in range(3):
        generate_sample("noprompt_pitch_ins.json",
                        prompt_idx=2,
                        print_new_events=True,
                        num_token_families=4,
                        out_dir=f"musictransformer/gen-noprompt{i}.mid",
                        save_prompt_separately=False,
                        **genconfig)
