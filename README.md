# musicllm
Attempt at creating a transformer capable of symbolic music generation. Focussing on creating a single drum track, piano (or keys equivalent) track and a bass.

## Proposed Architecture

## Data
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
### Download Datasets
Download the following datasets and save into working directory:
* My personal dataset: 
* LMD Cleaned: 

### Tokenize

### Prepare Dataset

### Train

### Generate
Download the model here and unzip into the working directory
* This model has been trained on 22 epochs - the folder will contain specific hyperparameters.
* Unfortunately the model's object name is called 'Transformer3' as I had experimented quite a few iterations beforehand

