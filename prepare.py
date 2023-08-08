import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# per https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb
class MIDIDataset(Dataset):
	"""
	Build Dataset from json files
		load_path: if loading a tensor into a dataset - should be .pt or .pth
		file_path: should be a directory containing json files
	"""

	def __init__(self, load_path: Union[str,Path]=None, block_size: int=512, file_path: Union[str,Path]=None, pad_token: int=0):

		samples = []

		# loading a saved tensor
		if load_path:
			if not isinstance(load_path, Path):
				load_path = Path(load_path)
			if not load_path.exists():
				raise FileNotFoundError('cannot find .pt or .pth file')
			
			self.samples = torch.load(load_path)
		
		else:

			if not block_size:
				raise ValueError('input block_size')

			if not isinstance(file_path, Path):
				file_path = Path(file_path)
			if not file_path.exists():
				raise FileNotFoundError('tokens folder not found')

			# Load json files and concatenate tokens
			for i in tqdm(file_path.glob("*.json")):
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
	
	def save(self, filename):
		if not filename:
			filename = "dataset.pt"
		torch.save(self.samples, filename)

def main():

	description = \
		"""
		Prepares Dataset object to train model. Does this by concatenating all the IDs together before splitting them into an N x block_size tensor.
		For simplicity, if the last row is not filled to block_size, it is removed.
		"""
	
	argparser = argparse.ArgumentParser("Create MIDIDataset object", description=description, usage="python prepare.py -fp path/to/tokens_folder -b 512")
	argparser.add_argument("-fp", "--file_path", default="tokens_pooled_pitch_ins", help="folder containing tokenized .json files (default: tokens_pooled_pitch_ins)")
	argparser.add_argument("-b", "--block_size", type=int, default=512, help="int object representing seq length of each row of data (default: 512)")
	argparser.add_argument("dataset_name", help="filename to save Dataset object for future loading - please include .pt suffix")

	args = argparser.parse_args()

	print(args)
	
	dataset = MIDIDataset(block_size=args.block_size, file_path=args.file_path, pad_token=0)
	print("Dataset size: ", dataset.samples.size())
	dataset.save(args.dataset_name)

if __name__ == "__main__":

	main()

