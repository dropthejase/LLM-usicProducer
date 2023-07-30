import json
from pathlib import Path
from pprint import pprint
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import MidiTokenizerPooled
from musictransformer import MusicTransformer

class Prompt(Dataset):
    """Allows use of Dataloader to move input_ids to cuda"""
    def __init__(self, ids: list[int]):
        self.data = torch.LongTensor(ids)

        if self.data.dim() == 1:
            self.data = self.data.reshape(1,-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}

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
    prompt = torch.LongTensor(testprompt[:prompt_idx]).view(1, -1, 4)

    print("Prompt Size: ", prompt.size())

    gen = model.generate(prompt, **kwargs)

    gen = gen.reshape(-1, 4).tolist()

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
    tokenizer = MidiTokenizerPooled()

    model = torch.load("musictransformer/musictransformer-full-2.pth")
    model.to(device)

    genconfig = {
        "temperature": [1.0, 1.0, 1.0, 1.0],
        "num_bars": 8,
        "max_steps": 512,
        "force_bar": False,
        "sampling_fn": "top_k",
        "threshold": [0.85, 0.85, 0.85, 0.85],
        "bar_token": 4
    }

    #generate_sample("tokens_pooled/0.json", prompt_idx=512, print_new_events=True, out_dir="musictransformer/gen-0.mid", save_prompt_separately=True, **genconfig)
    #generate_sample("tokens_pooled/16.json", prompt_idx=512, print_new_events=True, out_dir="musictransformer/gen-16.mid", save_prompt_separately=True, **genconfig)
    
    for i in range(3):
        generate_sample("noprompt.json", prompt_idx=512, print_new_events=True, out_dir=f"musictransformer/gen-noprompt{i}.mid", save_prompt_separately=False, **genconfig)

    