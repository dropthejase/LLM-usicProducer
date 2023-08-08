import argparse
import json
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Union

import torch

from musictransformer import MusicTransformer3
from tokenizer import MidiTokenizerPooled, MidiTokenizerNoPool

def generate_sample(json_file: Union[str,Path],
                    prompt_idx: int=None,
                    print_new_events: bool=False,
                    out_dir: Union[str,Path]=None,
                    save_prompt_separately: bool=True,
                    num_token_families: int=4,
                    **kwargs) -> list[int]:
    
    # if json_file given, load it, otherwise assume inference from scratch
    if json_file:
        with open(json_file) as jsonfile:
            testprompt = json.load(jsonfile)['ids']
        print("Number of tokens in prompt: ", len(testprompt))

        # check that prompt follows format required by transformer - i.e. T x 4 (pooled) or T x 1 (nopool)
        assert testprompt[0].__len__() == num_token_families, "prompt must be of format T x 4 (pooled) or T x 1 (nopool)"

    else:
        if num_token_families == 4:
            testprompt = [[1, 1, 1, 1]] # pooled
        elif num_token_families == 1:
            testprompt = [[1]] # nopool
        else:
            raise ValueError("num_token_families must be 1 (NoPool) or 4 (Pooled)")

    if not prompt_idx:
        prompt_idx = (len(testprompt) - 1) if len(testprompt) > 1 else 1 # take prompt_idx up to (but not including) EOS
    if prompt_idx > len(testprompt):
        prompt_idx = len(testprompt) - 1 # take prompt_idx up to (but not including) EOS
    prompt = torch.LongTensor(testprompt[:prompt_idx]).view(1, -1, num_token_families).to(device)

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


def main():

    device = torch.device("mps")

    # load model
    tokenizer = MidiTokenizerPooled()

    model = torch.load("musictransformer/musictransformer-full-22.pth")
    model.to(device)

    #[0.6, 0.8, 0.8, 0.5 or 0.6] or [0.6, 0.7, 0.7, 0.8] - best so far for 512 for musictransformer-full-7
    # higher temperatures at [0.7, 0.8, 0.8, 0.6-0.8] seem to work better for a more trained model like musictransformer-full-15
    # [0.8, 0.6-0.7, 0.6-0.7, 0.8] seems good for transformer-full-22
    genconfig = {
        "temperature": [0.8, 0.6, 0.6, 0.8], 
        "num_bars": 4,
        "max_steps": 512,
        "sampling_fn": "top_k",
        "threshold": [0.9, 0.9, 0.9, 0.9],
    }
  
    for i in tqdm(range(3)):
        generate_sample("noprompt_pitch_ins.json",
                        prompt_idx=2,
                        print_new_events=True,
                        num_token_families=4,
                        out_dir=f"musictransformer/gen-noprompt{i}.mid",
                        save_prompt_separately=False,
                        **genconfig)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Generate Samples", usage="TODO")
    argparser.add_argument("-p", "--prompt", default=None, help="Path to .json file containing prompt tokens. Leave blank for inference from scratch (default: None).")
    argparser.add_argument("-mp", "--model_path", default="musictransformer/musictransformer-full-22.pth", help="Path to .pth model")
    argparser.add_argument("-d", "--device", choices=["cpu","cuda","mps"], default="cpu", help="torch.device to use")
    argparser.add_argument("-pi", "--prompt_idx", help="Use if you want to truncate your prompt tokens")
    argparser.add_argument("-o", "--out_dir", default="generated_samples", help="Specify output folder for generated samples - include the '.mid' suffix (default: a folder called 'generated_samples')")
    argparser.add_argument("-sp", "--save_prompt", action="store_true", help="Use if you want to save your prompt separately for comparison purposes")
    argparser.add_argument("-pe", "--print_new_events", action="store_true", help="Use if you want to print new token events created")
    argparser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to generate")

    args = argparser.parse_args()

    # make output folder if it does not exist
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    # Load model and tokenizer
    device = torch.device(args.device)
    model = torch.load(args.model_path)
    model.to(device)

    if isinstance(model, MusicTransformer3):
        tokenizer = MidiTokenizerPooled()
        num_token_families = 4
    else:
        tokenizer = MidiTokenizerNoPool()
        num_token_families = 1

    # gen config
    genconfig = {
        "temperature": [0.8, 0.6, 0.6, 0.8], 
        "num_bars": 8,
        "max_steps": 512,
        "sampling_fn": "top_k",
        "threshold": [0.9, 0.9, 0.9, 0.9],
    }
    
    # generate
    for i in tqdm(range(args.num_samples)):
        generate_sample(json_file=args.prompt,
                        prompt_idx=args.prompt_idx,
                        print_new_events=args.print_new_events,
                        num_token_families=num_token_families,
                        out_dir=f"{args.out_dir}/{i}.mid",
                        save_prompt_separately=args.save_prompt,
                        **genconfig)