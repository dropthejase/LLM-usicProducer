# musicllm
Attempt at creating a transformer capable of symbolic music generation. Focussing on creating a single drum track, piano (or keys equivalent) track and a bass.

## Proposed Architecture



## Data
### My own dataset
A bunch of my own songs for fun!

### LA MIDI Dataset
The LA MIDI Dataset has been cleaned in the following ways:
* Removed songs without 4/4 time
* Removed songs with no drums
* Stripped midi such that there are only:
 * Three parts: drums, bass (programs 33-40) and piano (or piano equivalent)
 * The piano equivalent is based on a piano (programs 1-8) / guitar (25-32) / synth pads (89-96) / organs (17-24); if there are multiple candidates, then for simplicity, we take the instrument with the most notes

During tokenization, any further songs that create errors are removed. Typically this occurs when the MIDI has a pitch that is outside of the range of pitches available in our vocab list.

## Samples

## Other Tests

## Instructions
### Training
#### Download Datasets
Download the LMD dataset and save into working directory:
Note I have removed my own songs from this dataset - sorry!

#### Tokenize

#### Prepare Dataset

#### Train

### Generate
Download the model here and unzip into the working directory
* This model has been trained on 22 epochs - the folder will contain specific hyperparameters.
* Unfortunately the model's object name is called 'Transformer3' as I had experimented quite a few iterations beforehand

## Acknowledgments

## References
Dong, H.-W., Chen, K., Dubnov, S., McAuley, J., & Berg-Kirkpatrick, T. (2022). Multitrack Music Transformer: Learning Long-Term Dependencies in Music with Diverse Instruments. ArXiv:2207.06983 [Cs, Eess]. https://arxiv.org/abs/2207.06983
Hsiao, W.-Y., Liu, J.-Y., Yeh, Y.-C., & Yang, Y.-H. (2021). Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs. ArXiv:2101.02402 [Cs, Eess]. https://arxiv.org/abs/2101.02402