import json
from pathlib import Path
from pprint import pprint
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import MidiTokenizer, MidiTokenizer2, MidiTokenizer3, MidiTokenizer4
from transformers import GPT2LMHeadModel, GPT2Config, GenerationConfig

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

def generate_sample(json_file: Union[str,Path], prompt_idx: int=500, print_new_events: bool=False, out_dir: Union[str,Path]=None, save_prompt_separately: bool=True) -> list[int]:
    with open(json_file) as jsonfile:
        testprompt = json.load(jsonfile)['ids']
    print("Length of original JSON file: ", len(testprompt))

    prompt = Prompt(testprompt[:prompt_idx])

    dataloader_test = DataLoader(prompt)

    for i in dataloader_test:
        gen = model.generate(i['input_ids'].to(model.device), generation_config=generation_config)

    gen = gen.reshape(-1).tolist()

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
    tokenizer = MidiTokenizer4()

    model = torch.load('gpt2/gpt2-full-0.pth')
    model.to(device)
    model.eval()

    # generation
    generation_config = GenerationConfig(
        min_new_tokens=512,
        max_length=2048,
        temperature=0.9, # was 0.5
        num_beams=1,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        epsilon_cutoff=3e-4,
        eta_cutoff=1e-3,
        bos_token_id=tokenizer.vocab['BOS_None'],
        pad_token_id=tokenizer.vocab['PAD_None'],
    )

    #for name, param in model.named_parameters():
    #  print(name, param.size())
    gen = generate_sample("tokens4/12.json", 512, print_new_events=True, out_dir='gpt2/gen-song16.mid', save_prompt_separately=True)
    gen = generate_sample("tokens4/13.json", 512, print_new_events=True, out_dir='gpt2/gen-song20.mid', save_prompt_separately=True)
    gen = generate_sample("tokens4/14.json", 512, print_new_events=True, out_dir='gpt2/gen-song24.mid', save_prompt_separately=True)
    gen = generate_sample("gpt2/noprompt.json", 512, print_new_events=True, out_dir='gpt2/gen-noprompt.mid', save_prompt_separately=True)