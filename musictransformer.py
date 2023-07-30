from x_transformers.x_transformers import Decoder
from x_transformers.autoregressive_wrapper import top_k, top_p

import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import MidiTokenizerPooled

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MusicTransformerWrapper(nn.Module):
    def __init__(self,
                 n_tokens: list[int],
                 emb_sizes: list[int],
                 n_layers=12,
                 n_heads=8,
                 d_model=512,
                 dropout=0.1):
        """
        n_tokens: encoding["n_tokens"] shows vocab size of each token family
        emb_sizes: size of embeddings for each token family in order of start_pos, note_dur, pitch, ins
        n_layer: how many transformer blocks
        n_heads: how many attention heads
        """
        super().__init__()

        # params
        self.n_tokens = n_tokens
        self.emb_sizes = emb_sizes

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

        self.loss_fn = nn.CrossEntropyLoss()

        # embeddings
        self.emb_start_pos = nn.Embedding(self.n_tokens[0], self.emb_sizes[0])
        self.emb_note_dur = nn.Embedding(self.n_tokens[1], self.emb_sizes[1])
        self.emb_pitch = nn.Embedding(self.n_tokens[2], self.emb_sizes[2])
        self.emb_instrument = nn.Embedding(self.n_tokens[3], self.emb_sizes[3])

        # project to d_model after concatenating in forward()
        # N x T x sum(emb_sizes) -> N x T x d_model
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # fast_transformer - output is N x T x d_model
        self.transformer_encoder = Decoder(
            depth = self.n_layers,
            heads = self.n_heads,
            dim = self.d_model,
            layer_dropout = self.dropout,
            attn_dropout =self.dropout,
            ff_dropout = self.dropout)

        # to logits - N x T x d_model -> token_family_size for n in n_tokens
        self.logit_start_pos = nn.Linear(self.d_model, self.n_tokens[0])
        self.logit_note_dur = nn.Linear(self.d_model, self.n_tokens[1])
        self.logit_pitch = nn.Linear(self.d_model, self.n_tokens[2])
        self.logit_instrument = nn.Linear(self.d_model, self.n_tokens[3])

    def forward(self, x, show_sizes=False, mask=None):

        # embeddings
        # N x (T x 4) -> embed( N x (T x 4[i]) ) for i in each token family
        # outputs N x T x emb_size

        emb_start_pos = self.emb_start_pos(x[..., 0])
        emb_note_dur = self.emb_note_dur(x[..., 1])
        emb_pitch = self.emb_pitch(x[..., 2])
        emb_instrument = self.emb_instrument(x[..., 3])

        if show_sizes:
            print("Embeddings: ", emb_start_pos.size())

        # concat -> N x T x sum(embed_size)
        embs = torch.cat([emb_start_pos,
                          emb_note_dur,
                          emb_pitch,
                          emb_instrument], dim=-1)

        if show_sizes:
            print("Concat Embeddings: ", embs.size())

        # project N x T x sum(emb_size) -> N x T x d_model
        embed_linear = self.in_linear(embs)

        if show_sizes:
            print("Projected to N x T x d-Model: ", embs.size())

        # add positional encoding TO DO
        to_trans = self.pos_encoding(embed_linear)

        # transformer blocks -> N x T x d_model
        h = self.transformer_encoder(to_trans, mask=None)

        if show_sizes:
            print("Output should be (N x T x d_model): ", h.size())

        # project back to individual logits N x T x d_model -> N x T x n_tokens
        y_start_pos = self.logit_start_pos(h)
        y_note_dur = self.logit_note_dur(h)
        y_pitch = self.logit_pitch(h)
        y_instrument = self.logit_instrument(h)

        if show_sizes:
            print("start_pos logits: ", y_start_pos.size())
            print("note_dur logits: ", y_note_dur.size())
            print("pitch logits: ", y_pitch.size())
            print("instrument logits: ", y_instrument.size())

        # shape = 4 x N x T x n_tokens
        return y_start_pos, y_note_dur, y_pitch, y_instrument

    def compute_loss(self, outputs, targets):

        # need to transpose N x T x n_tokens -> N x n_tokens x T for CrossEntropyLoss
        losses = [self.loss_fn(outputs[i].transpose(2,1),
                               targets[..., i]) for i in range(len(outputs))]

        # calculate average loss
        loss = sum(losses) / 4

        return loss

class MusicTransformer(nn.Module):
    def __init__(self,
                 n_tokens: list[int],
                 emb_sizes: list[int],
                 n_layers=12,
                 n_heads=8,
                 d_model=512,
                 dropout=0.1):

        super().__init__()

        self.model = MusicTransformerWrapper(
                    n_tokens=n_tokens,
                    emb_sizes=emb_sizes,
                    n_layers=12,
                    n_heads=8,
                    d_model=512,
                    dropout=0.1)

    def forward(self, x, show_sizes=False, mask=None):
        return self.model(x, show_sizes, mask)

    def compute_loss(self, outputs, targets):
        return self.model.compute_loss(outputs, targets)

    def save_params(self, filepath):
        config = {
            "n_tokens": self.model.n_tokens,
            "emb_sizes": self.model.emb_sizes,
            "n_layers": self.model.n_layers,
            "n_heads": self.model.n_heads,
            "d_model": self.model.d_model,
            "dropout p": self.model.dropout
        }

        with open(filepath, "w") as file:
            json.dump(config, file)
    
    @torch.no_grad()
    def generate(self, prompt, temperature=[1.2, 2.0, 0.9, 1.2], num_bars=8,  max_steps=50, force_bar=False, sampling_fn="top_k", threshold=[1.0, 0.9, 0.9, 0.9], bar_token=4):
        """
        Generates samples
            prompt: should be a LongTensor of size N x T x 4 (for each of the token families)
            temperature: list of 4 floats corresponding to each of the token families in order start_pos, note_dur, pitch, instrument
            num_bars: number of bars to generate
            max_steps: max number of tokens (i.e. Notes) to be generated (in case model doesn't learn bars) and there's an inf loop. This is the 'absolute' stopper.
            force_bar: if any token has bar token, all of them will have it too; i.e. this "forces" a bar
            threshold: list of 4 floats corresponding to each of the token families in order start_pos, note_dur, pitch, instrument
        """

        self.model.eval()

        bar_count = 1
        steps = 0

        while bar_count < num_bars and steps < max_steps:
            outputs = self.model(prompt)
            
            # extract last logit for each token family -> 1D LongTensor
            logit_start_pos = outputs[0][:, -1, :]
            logit_note_dur = outputs[1][:, -1, :]
            logit_pitch = outputs[2][:, -1, :]
            logit_instrument = outputs[3][:, -1, :]

            # sample
            sample_start_pos = self.sample(logit_start_pos, temperature=temperature[0], sampling_fn=sampling_fn, threshold=threshold[0])
            sample_note_dur = self.sample(logit_note_dur, temperature=temperature[1], sampling_fn=sampling_fn, threshold=threshold[1])
            sample_pitch = self.sample(logit_pitch, temperature=temperature[2], sampling_fn=sampling_fn, threshold=threshold[2])
            sample_instrument = self.sample(logit_instrument, temperature=temperature[3], sampling_fn=sampling_fn, threshold=threshold[3])

            # add bars
            if (sample_start_pos == bar_token) or (sample_note_dur == bar_token) or (sample_pitch == bar_token) or (sample_instrument == bar_token):
                # force a bar if force_bar = True
                if force_bar:
                    sample_start_pos = torch.LongTensor(4)
                    sample_note_dur = torch.LongTensor(4)
                    sample_pitch = torch.LongTensor(4)
                    sample_instrument = torch.LongTensor(4)
                    bar_count += 1

                # else if there's a bar token, skip loop if other token families don't also output bar token
                else:
                    if sum([sample_start_pos.item(), sample_note_dur.item(), sample_pitch.item(), sample_instrument.item()]) != (4 * bar_token):
                        continue
                    else:
                        bar_count += 1
                        print(f"Created {bar_count - 1} bars")

            # concat 4 x 1D LongTensor -> (1, 1, 4)
            pred_ids = torch.concat([
                sample_start_pos,
                sample_note_dur,
                sample_pitch,
                sample_instrument
            ]).view(1, 1, -1)

            prompt = torch.hstack((prompt, pred_ids))

            steps += 1

        return prompt

    def sample(self, logits, temperature, sampling_fn="top_k", threshold=0.9):
        if sampling_fn == "top_k":
            probs = F.softmax(top_k(logits, thres=threshold) / temperature, dim=-1)
        elif sampling_fn == "top_p":
            probs = F.softmax(top_p(logits, thres=threshold) / temperature, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1)

if __name__ == "__main__":

    test = torch.LongTensor([[[0, 0, 0, 0],
                   [4, 4, 4, 4]]])
    
    tokenizer = MidiTokenizerPooled()

    model = MusicTransformer(tokenizer.vocab['n_tokens'], [64, 256, 256, 16], n_layers=6)
    print_trainable_parameters(model)
    
    out = model(test)
    for logit in out:
        print(logit.size())
