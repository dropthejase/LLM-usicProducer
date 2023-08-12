import argparse
from datetime import datetime
import json
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Union

import torch

from musictransformer import MusicTransformer3, MusicTransformerXL
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
    print("Start generating from idx: ", prompt_idx)
    print(prompt)

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

    usage =\
        """
        \n
        Example: generate 3 samples from scratch (leave --prompt blank for inference from scratch), saved in an 'inferences_from_scratch' folder

            -mp path/to/model.pth -d mps -o "inferences_from_scratch" -n 3

        Example: generate 1 sample from prompt

            -p path/to/prompt.json -mp path/to/model.pth -d mps -o "generated"

        """

    argparser = argparse.ArgumentParser("Generate Samples", usage=usage)

    argparser.add_argument("-p", "--prompt", default=None, help="Path to .json file containing prompt tokens. Leave blank for inference from scratch (default: None).")
    argparser.add_argument("-mp", "--model_path", default="musictransformer/musictransformer-full-22.pth", help="Path to .pth model")
    argparser.add_argument("-d", "--device", choices=["cpu","cuda","mps"], default="cpu", help="torch.device to use")
    argparser.add_argument("-pi", "--prompt_idx", type=int, help="Use if you want to truncate your prompt tokens")
    argparser.add_argument("-o", "--out_dir", default="generated_samples", help="Specify output folder for generated samples - include the '.mid' suffix (default: a folder called 'generated_samples')")
    argparser.add_argument("-sp", "--save_prompt", action="store_true", help="Use if you want to save your prompt separately for comparison purposes")
    argparser.add_argument("-pe", "--print_new_events", action="store_true", help="Use if you want to print new token events created")
    argparser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to generate")

    argparser.add_argument("--genconfig", help="path to .json file containing genconfig")

    args = argparser.parse_args()
    print(args)

    # make output folder if it does not exist
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    # Load model and tokenizer
    device = torch.device(args.device)
    model = torch.load(args.model_path)
    model.to(device)

    if isinstance(model, MusicTransformer3) or isinstance(model, MusicTransformerXL):
        tokenizer = MidiTokenizerPooled()
        num_token_families = 4
        TEMPERATURE = [0.8, 0.6, 0.6, 0.8]
        THRESHOLD = [0.9, 0.9, 0.9, 0.9]

    else:
        tokenizer = MidiTokenizerNoPool()
        num_token_families = 1
        TEMPERATURE = [0.9]
        THRESHOLD = [0.9]


    # gen config
    #[0.6, 0.8, 0.8, 0.5 or 0.6] or [0.6, 0.7, 0.7, 0.8] - best so far for 512 for musictransformer-full-7
    # higher temperatures at [0.7, 0.8, 0.8, 0.6-0.8] seem to work better for a more trained model like musictransformer-full-15
    # [0.8, 0.6-0.7, 0.6-0.7, 0.8] seems good for transformer-full-22

    # defaults
    genconfig = {
            "temperature": TEMPERATURE, 
            "num_bars": 32,
            "max_steps": 256,
            "sampling_fn": "top_k",
            "threshold": THRESHOLD,
        }

    # if genconfig json file provided, take those values where available
    if args.genconfig:
        with open(args.genconfig) as jsonfile:
            config = json.load(jsonfile)

        assert config["temperature"].__len__() == num_token_families, "Temperature must be list[float] where len(list) is 4 if using pooled embeddings or 1 otherwise"
        assert config["threshold"].__len__() == num_token_families, "Threshold must be list[float] where len(list) is 4 if using pooled embeddings or 1 otherwise"

        # override defaults where applicable
        for k, v in config.items():
            genconfig[k] = v
    
    # log gen config
    with open(f"{out_dir}/genconfigs.txt", "a") as f:
        f.write("============================================================================================================================================ \n")
        f.write(f"{datetime.now()}\n")
        f.write("============================================================================================================================================ \n")
        f.write(f"{genconfig}\n")
        f.write(f"\n JSON_FILE: {'inference from scratch' if not args.prompt else args.prompt}")
        f.write(f"\n PROMPT_IDX: {args.prompt_idx}")
        f.write(f"\n OUTPUT_LOCATION: {args.out_dir}")
        f.write(f"\n")
        f.write(f"\n")
        
    # generate
    for i in tqdm(range(args.num_samples)):
        generate_sample(json_file=args.prompt,
                        prompt_idx=args.prompt_idx,
                        print_new_events=args.print_new_events,
                        num_token_families=num_token_families,
                        out_dir=f"{args.out_dir}/{i}.mid",
                        save_prompt_separately=args.save_prompt,
                        **genconfig)
