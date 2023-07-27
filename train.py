from tokenizer import MidiTokenizer, MidiTokenizer2, MidiTokenizer3, MidiTokenizer4
from prepare import MIDIDataset

from tqdm import tqdm
from pathlib import Path
from typing import Union

import json

from pprint import pprint

import numpy as np

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

import evaluate
from transformers import GPT2LMHeadModel, GPT2Config, TransfoXLConfig, TransfoXLLMHeadModel, Trainer, TrainingArguments, get_scheduler

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

def train(train_dataloader, model, num_epochs=3, lr=5e-4, filename="model", loggingsteps=1000):

    # print num parameters
    print_trainable_parameters(model)

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
    dataset = MIDIDataset(block_size=512, file_path='tokens4', pad_token=tokenizer.vocab['PAD_None'])
    print(dataset.samples.size())
    
    # split data
    train_dataset, eval_dataset = random_split(dataset, [0.8, 0.2])

    # create dataloaders
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    ### create model - GPT2 ###
    '''config = GPT2Config(
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

    train(train_dataloader=train_dataloader, model=model, num_epochs=3, lr=5e-4, filename="transXL", loggingsteps=1000)'''

    ### create model - TransfoXL - note need to use Trainer ###
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions.flatten(), references=labels.flatten())

    training_args = TrainingArguments(
        "transXL",
        evaluation_strategy="no",
        logging_strategy="steps",
        logging_steps=25000,
        weight_decay=0.01,
        warmup_ratio=0.3,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        optim="adamw_torch",
        #gradient_checkpointing=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=5000,
        # eval_accumulation_steps=500,
        #fp16=True,
    )

    config = TransfoXLConfig(
        vocab_size=len(tokenizer.vocab),
        cutoffs=[],
        d_model=512,
        d_head=512//8,
        d_embd=512,
        d_inner=2048,
        n_layer=4,
        n_head=8,
        padding_token_id=tokenizer.vocab['PAD_None'],
        bos_token_id=tokenizer.vocab['BOS_None'],
        eos_token_id=tokenizer.vocab['EOS_None'],
    )

    model = TransfoXLLMHeadModel(config)
    model.config.save_pretrained("transXL")
    print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
 


    