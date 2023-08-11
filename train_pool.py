from tokenizer import MidiTokenizerPooled, MidiTokenizerNoPool
from prepare import MIDIDataset
from musictransformer import MusicTransformer3


import argparse
import csv
from pathlib import Path
from tqdm import tqdm
from typing import Union

import json
from pprint import pprint

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

    return trainable_params, all_param


def train(x, model, batch_size=8, num_epochs=3, lr=1e-4, filename="model", loggingsteps=1000):
    """
    x: a Dataset object comprising training dataset
    """
    # make folder
    if not isinstance(filename,Path):
        filename = Path(filename)
    if not filename.exists():
        filename.mkdir()
    
    # save hyperparameters
    model.save_params(f"{filename}/hyperparameters.json")

    # add num_parameters to hyperparameters
    with open(f"{filename}/hyperparameters.json") as jsonfile:
        hyperparams = json.load(jsonfile)
    
    trainable_params, all_params = print_trainable_parameters(model)
    hyperparams["trainable parameters"] = trainable_params
    hyperparams["all parameters"] = all_params
    
    with open(f"{filename}/hyperparameters.json", "w") as jsonfile:
        json.dump(hyperparams, jsonfile)

    # prepare train_losses.csv
    with open(f"{filename}/train_losses.csv", "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['steps', 'train_loss'])
        writer.writeheader()

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr) 

    num_batch = len(x) // batch_size

    # create dataloader
    train_dataloader = DataLoader(x, batch_size=batch_size, shuffle=True)

    # training loop
    steps = 0
    for epoch in range(num_epochs):   
        
        losses = []
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}"):

            optimizer.zero_grad()

            batch = batch["input_ids"]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = model.compute_loss(outputs, targets)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if steps % loggingsteps == 0:
                print(f"Step: {steps}, Train Loss: {loss.item():.4f}")
                torch.save(model, f"{filename}/{filename}-checkpoint-{epoch}-{steps}.pth")

                # log to csv
                with open(f"{filename}/train_losses.csv", "a") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['steps', 'train_loss'])
                    writer.writerow({'steps': steps, 'train_loss': loss.item()})
                
            steps += batch_size

        print(f"Epoch {epoch} Average Loss: {sum(losses) / num_batch}")
        torch.save(model, f"{filename}/{filename}-full-{epoch}.pth")


if __name__ == "__main__":

    argsparser = argparse.ArgumentParser("Trainer", description="To train the model", usage="TODO")
    argsparser.add_argument("-t", "--tokenizer", choices=["pooled", "nopool"], default="pooled", help="Choose between 'pooled' (MidiTokenizerPooled) or 'nopool' (MidiTokenizerNoPool) (default: pooled)")
    argsparser.add_argument("-fp", "--from_pretrained", help="Path to pretrained model or checkpoint .pth file")
    argsparser.add_argument("--dataset_path", default="dataset_pitch_ins512.pt", help="Path to Dataset Object .pt file (default: dataset_pitch_ins512.pt)")
    argsparser.add_argument("--train_split", type=float, default=0.9, help="Train split (default: 0.9)")
    argsparser.add_argument("-d", "--device", choices=["cpu", "cuda", "mps"], default="mps", help="Choose torch.device")

    # model configs
    argsparser.add_argument("--model_config", help="Path to model_config.json file if exists or applicable")

    # training configs
    argsparser.add_argument("--training_args", help="Path to training_args.json file if exists or applicable")

    args = argsparser.parse_args()
    print(vars(args))

    if args.tokenizer == "nopool":
        tokenizer = MidiTokenizerNoPool()
    else:
        tokenizer = MidiTokenizerPooled()

    # load Dataset
    dataset = MIDIDataset(load_path=args.dataset_path)
    print("Dataset size: ", dataset.samples.size())
    
    # split data
    train_dataset, eval_dataset = random_split(dataset, [args.train_split, 1 - args.train_split])

    if args.from_pretrained:
        # load model
        model = torch.load(args.from_pretrained)
    
    else:
        model_config = {"emb_sizes": [512, 128, 256, 512],
                "emb_pooling": "concat",
                "n_layers": 12,
                "n_heads": 8,
                "d_model": 512,
                "dropout": 0.1
        }
        
        if args.model_config:
            with open(args.model_config) as jsonfile:
                config = json.load(jsonfile)
            for k, v in config.items():
                model_config[k] = v
        
        # create model
        model = MusicTransformer3(
            n_tokens=tokenizer.vocab["n_tokens"],
            **model_config
        )            

    device = torch.device(args.device)
    model.to(device)

    print("Train Dataset Size: ", len(train_dataset))

    training_args = {"batch_size": 12,
                    "num_epochs": 1,
                    "lr": 5e-4,
                    "filename": "musictransformerXL",
                    "loggingsteps": 10000}
    
    if args.training_args:
        with open(args.training_args) as jsonfile:
            t_args = json.load(jsonfile)
        for k, v in t_args.items():
            training_args[k] = v
    
    train(train_dataset, model, **training_args)

 


    