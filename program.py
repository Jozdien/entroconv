from diarize import transcript
from gpt2 import surprise

import pprint
pp = pprint.PrettyPrinter(indent=4)

transcript = transcript(AUDIO_FILENAME="audio.wav", num_speakers=2)
for text in transcript:
	start = text.index(":", text.find("speaker")) + 2
	print(text[start:])
pp.pprint(transcript)


'''

TODO:

Write program file that takes in transcript as input and returns a list of sets of the form (speaker_name, text)

Write function that takes a set as input and calculates entropy for each word.
	- For this, keep a running prefix string containing the entire conversation so far, which will be the first parameter to surprise()
	- Everytime a new set is taken, append "speaker_name: " to the running prefix string and keep going

'''