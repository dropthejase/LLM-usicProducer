from miditoolkit.midi import parser
from miditoolkit.midi import containers as ct

file = parser.MidiFile("test/song0.mid")
print(file)