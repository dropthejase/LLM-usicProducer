from tokenizer import MidiTokenizer, MidiTokenizer2, MidiTokenizer3, MidiTokenizer4
from prepare import MIDIDataset

from tqdm import tqdm
from pathlib import Path
from typing import Union

import json

from pprint import pprint

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, get_scheduler


def train(train_dataloader, model, num_epochs=3, lr=5e-4, filename="model", loggingsteps=1000):

    # make folder
    if not isinstance(filename,Path):
        filename = Path(filename)
    if not filename.exists():
        filename.mkdir()

    # save config
    model.config.save_pretrained(filename)
    
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr) 

    # create scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name='cosine_with_restarts',optimizer=optimizer, num_warmup_steps=1e2, num_training_steps=num_training_steps)

    # training loop
    model.train()
    for epoch in range(num_epochs):   
        steps = 0
        for batch in tqdm(train_dataloader):

            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            lr_scheduler.step()
            optimizer.step()

            if steps % loggingsteps == 0:
                print(f"Step: {steps}, Train Loss: {loss:.4f}")
                torch.save(model, f"{filename}/{filename}-checkpoint-{epoch}-{steps}.pt")
                
            steps += 1

        torch.save(model, f"{filename}/{filename}-full-{epoch}.pth")

    print(f"Final Train Loss: {loss:.4f}")

if __name__ == "__main__":
    tokenizer = MidiTokenizer4()

    # create Dataset
    dataset = MIDIDataset(block_size=1024, file_path='tokens4', pad_token=tokenizer.vocab['PAD_None'])
    print(dataset.samples.size())
    
    # split data
    train_dataset, eval_dataset = random_split(dataset, [0.9, 0.1])

    # create dataloaders
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    # create model
    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=2048,
        n_embd=512,
        n_layer=6,
        n_head=8,
        padding_token_id=tokenizer.vocab['PAD_None'],
        bos_token_id=tokenizer.vocab['BOS_None'],
        eos_token_id=tokenizer.vocab['EOS_None'],
    )

    model = GPT2LMHeadModel(config)

    device = torch.device("mps")
    model.to(device)

    train(train_dataloader=train_dataloader, model=model, num_epochs=3, lr=5e-4, filename="gpt2", loggingsteps=1000)
 


    