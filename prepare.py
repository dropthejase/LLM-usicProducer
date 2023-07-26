import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# per https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb
class MIDIDataset(Dataset):
  """Build Dataset from json files"""

  def __init__(self, block_size: int, file_path: Union[str,Path], pad_token: int=0):

    samples = []

    if not isinstance(file_path, Path):
      file_path = Path(file_path)
    if not file_path.exists():
      raise FileNotFoundError('tokens folder not found')

    # Load json files and concatenate tokens
    for i in file_path.glob("*.json"):
      with open(i) as json_file:
        tokens = json.load(json_file)['ids']
        samples += tokens

    # split by block size
    self.total_tokens = len(samples)

    N = self.total_tokens // block_size

    self.samples = [torch.LongTensor(samples[i*block_size:(i+1)*block_size]) for i in range(N)]
    #self.samples.append(torch.LongTensor(samples[N*block_size:])) - truncate last bit

    # pad
    self.samples = pad_sequence(self.samples, batch_first=True, padding_value=pad_token)

  def __getitem__(self, idx):
    return {"input_ids": self.samples[idx], "labels": self.samples[idx]}

  def __len__(self):
    return len(self.samples)