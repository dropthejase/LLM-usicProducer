from tokenizer import MidiTokenizerPooled
from prepare import MIDIDataset
from musictransformer import MusicTransformer

from tqdm import tqdm
from pathlib import Path
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

def train(x, model, batch_size=8, num_epochs=3, lr=1e-4, filename="model", loggingsteps=1000):
    """
    x: a Dataset object comprising training dataset
    """

    # print num parameters
    print_trainable_parameters(model)

    # make folder
    if not isinstance(filename,Path):
        filename = Path(filename)
    if not filename.exists():
        filename.mkdir()
    
    # save hyperparameters
    model.save_params(f"{filename}/hyperparameters.json")

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
        for batch in tqdm(train_dataloader):

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
                torch.save(model, f"{filename}/{filename}-checkpoint-{epoch}-{steps}.pt")
                
            steps += batch_size

        print(f"Epoch {epoch} Average Loss: {sum(losses) / num_batch}")
        torch.save(model, f"{filename}/{filename}-full-{epoch}.pth")

    print(f"Final Train Loss: {loss:.4f}")

if __name__ == "__main__":
    tokenizer = MidiTokenizerPooled()

    # load Dataset
    dataset = MIDIDataset(load_path="dataset.pt")
    print("Dataset size: ", dataset.samples.size())
    
    # split data
    train_dataset, eval_dataset = random_split(dataset, [0.9, 0.1])

    # create model
    model = MusicTransformer(
        n_tokens=tokenizer.vocab["n_tokens"],
        emb_sizes=[64, 256, 512, 8],
        n_layers=12,
        n_heads=8,
        d_model=512,
        dropout=0.1)

    device = torch.device("mps")
    model.to(device)

    train(train_dataset, model, batch_size=8, num_epochs=3, lr=1e-4, filename="musictransformer", loggingsteps=20000)

 


    