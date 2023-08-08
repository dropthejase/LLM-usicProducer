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


class MusicTransformerWrapper3(nn.Module):
    """Uses pitch_instrument encodings, produces 1 x 4 tokens"""
    def __init__(self,
                 n_tokens: list[int],
                 emb_sizes: list[int],
                 emb_pooling: str,
                 n_layers=12,
                 n_heads=8,
                 d_model=512,
                 dropout=0.1):
        """
        n_tokens: encoding["n_tokens"] shows vocab size of each token family
        emb_sizes: size of embeddings for each token family in order of start_pos, note_dur, pitch, ins
        emb_pooling: 'concat' or 'sum'
        n_layer: how many transformer blocks
        n_heads: how many attention heads
        """
        super().__init__()

        # params
        self.n_tokens = n_tokens
        self.emb_sizes = emb_sizes
        
        if emb_pooling != "concat" and emb_pooling != "sum":
            raise ValueError("emb_pooling must be 'concat' or 'sum")
        if emb_pooling == "sum" and len(set(emb_sizes)) != 1:
            raise ValueError("if emb_pooling is 'sum', then emb_sizes must be same value")
        self.emb_pooling = emb_pooling

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

        self.loss_fn = nn.CrossEntropyLoss()

        # embeddings
        self.emb_bar = nn.Embedding(self.n_tokens[0], self.emb_sizes[0])
        self.emb_start_pos = nn.Embedding(self.n_tokens[1], self.emb_sizes[1])
        self.emb_note_dur = nn.Embedding(self.n_tokens[2], self.emb_sizes[2])
        self.emb_pitch_instrument = nn.Embedding(self.n_tokens[3], self.emb_sizes[3])

        # project to d_model after concatenating in forward()
        # N x T x sum(emb_sizes) or N x T x emb_size[0] -> N x T x d_model
        if self.emb_pooling == "sum" and self.emb_sizes[0] == self.d_model:
            self.in_linear = nn.Identity()
        else:
            self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

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
        self.logit_bar = nn.Linear(self.d_model, self.n_tokens[0])
        self.logit_start_pos = nn.Linear(self.d_model, self.n_tokens[1])
        self.logit_note_dur = nn.Linear(self.d_model, self.n_tokens[2])
        self.logit_pitch_instrument = nn.Linear(self.d_model, self.n_tokens[3])

    def forward(self, x, show_sizes=False, mask=None):

        # embeddings
        # N x (T x 4) -> embed( N x (T x 4[i]) ) for i in each token family
        # outputs N x T x emb_size

        emb_bar = self.emb_bar(x[..., 0])
        emb_start_pos = self.emb_start_pos(x[..., 1])
        emb_note_dur = self.emb_note_dur(x[..., 2])
        emb_pitch_instrument = self.emb_pitch_instrument(x[..., 3])

        if show_sizes:
            print("Embeddings: ", emb_start_pos.size())

        if self.emb_pooling == "concat":
            embs = torch.cat([emb_bar,
                            emb_start_pos,
                            emb_note_dur,
                            emb_pitch_instrument], dim=-1)
        else:
            embs = sum((emb_bar,
                       emb_start_pos,
                       emb_note_dur,
                       emb_pitch_instrument))

        if show_sizes:
            print("Pooled Embeddings: ", embs.size())

        # project N x T x sum(emb_size) -> N x T x d_model
        embed_linear = self.in_linear(embs)
        embed_linear = self.layer_norm(embed_linear)

        if show_sizes:
            print("Projected to N x T x d-Model: ", embs.size())

        # add positional encoding TO DO
        to_trans = self.pos_encoding(embed_linear)

        # transformer blocks -> N x T x d_model
        h = self.transformer_encoder(to_trans, mask=None)

        if show_sizes:
            print("Output should be (N x T x d_model): ", h.size())

        # project back to individual logits N x T x d_model -> N x T x n_tokens
        y_bar = self.logit_bar(h)
        y_start_pos = self.logit_start_pos(h)
        y_note_dur = self.logit_note_dur(h)
        y_pitch_instrument = self.logit_pitch_instrument(h)

        if show_sizes:
            print("bar: ", y_bar.size())
            print("start_pos logits: ", y_start_pos.size())
            print("note_dur logits: ", y_note_dur.size())
            print("pitch logits: ", y_pitch_instrument.size())

        # shape = 5 x N x T x n_tokens
        return y_bar, y_start_pos, y_note_dur, y_pitch_instrument

    def compute_loss(self, outputs, targets):

        # need to transpose N x T x n_tokens -> N x n_tokens x T for CrossEntropyLoss
        losses = [self.loss_fn(outputs[i].transpose(2,1),
                               targets[..., i]) for i in range(len(outputs))]

        # calculate average loss
        loss = sum(losses) / len(losses)

        return loss


class MusicTransformer3(nn.Module):
    def __init__(self,
                 n_tokens: list[int],
                 emb_sizes: list[int],
                 emb_pooling: str="concat",
                 n_layers=12,
                 n_heads=8,
                 d_model=512,
                 dropout=0.1):

        super().__init__()

        self.model = MusicTransformerWrapper3(
                    n_tokens=n_tokens,
                    emb_sizes=emb_sizes,
                    emb_pooling=emb_pooling,
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
            "emb_pooling": self.model.emb_pooling,
            "n_layers": self.model.n_layers,
            "n_heads": self.model.n_heads,
            "d_model": self.model.d_model,
            "dropout p": self.model.dropout
        }

        with open(filepath, "w") as file:
            json.dump(config, file)
    
    @torch.no_grad()
    def generate(self, prompt, temperature=[1.0, 1.0, 1.0, 1.0], num_bars=8,  max_steps=50, sampling_fn="top_k", threshold=[0.9, 0.9, 0.9, 0.9]):
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

        # if last tokens are EOS, take the next last
        if prompt[:, -1, :].view(-1)[0].item() == 2:
            current_bar = prompt[:, -2, :].view(-1)[0].item()
        elif prompt[:, -1, :].view(-1)[0].item() == 1:
            current_bar = 3
        else:
            current_bar = prompt[:, -1, :].view(-1)[0].item()
        
        steps = 0
        start_bar = current_bar
        
        while current_bar <= (start_bar + num_bars) and steps < max_steps:
            outputs = self.model(prompt)
            
            
            # extract last logit for each token family -> 1D LongTensor
            logit_bar = outputs[0][:, -1, :]
            logit_start_pos = outputs[1][:, -1, :]
            logit_note_dur = outputs[2][:, -1, :]
            logit_pitch_instrument = outputs[3][:, -1, :]

            # make sure we are continuing the song - bar must be at least the same bar as 'current_bar'
            # also prevent 'skipping' more than 1 bars max
            sample_bar = self.sample(logit_bar, temperature=temperature[0], sampling_fn=sampling_fn, threshold=threshold[0])
            #print("step: ", steps, " sample bar: ", sample_bar.item(), " current bar: ", current_bar)

            '''while sample_bar.item() < current_bar:
                sample_bar = self.sample(logit_bar, temperature=temperature[0], sampling_fn=sampling_fn, threshold=threshold[0])'''
            '''while sample_bar.item() - current_bar > 1:
                print("step: ", steps, " sample bar: ", sample_bar.item(), " current bar: ", current_bar)
                sample_bar = self.sample(logit_bar, temperature=temperature[0], sampling_fn=sampling_fn, threshold=threshold[0])'''

            # sample the rest
            sample_start_pos = self.sample(logit_start_pos, temperature=temperature[1], sampling_fn=sampling_fn, threshold=threshold[1])
            sample_note_dur = self.sample(logit_note_dur, temperature=temperature[2], sampling_fn=sampling_fn, threshold=threshold[2])
            sample_pitch_instrument = self.sample(logit_pitch_instrument, temperature=temperature[3], sampling_fn=sampling_fn, threshold=threshold[3])

            # update bar_count (goes from token_id 4 onwards)
            if (sample_bar.item() - current_bar) > 0:
                current_bar += 1

            # concat 5 x 1D LongTensor -> (1, 1, 5)
            pred_ids = torch.concat([
                sample_bar,
                sample_start_pos,
                sample_note_dur,
                sample_pitch_instrument
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
    
    tokenizer = MidiTokenizerPooled()

    model = MusicTransformer3(tokenizer.vocab['n_tokens'], [512, 128, 256, 512], emb_pooling="concat", n_layers=6)
    print_trainable_parameters(model)
    
    test = torch.randint(1, 5, (1, 2, 4))

    out = model(test)
    for logit in out:
        print(logit.size())
    
    device = torch.device("mps")
    model = torch.load("musictransformer/musictransformer-full-22.pth").to(device)
    prompt = torch.randint(1, 5, (1, 2, 4)).to(device)
    prompt_res = model.generate(prompt, max_steps=3)
    print(prompt_res)
