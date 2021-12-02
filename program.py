from diarize import transcript

import pprint
pp = pprint.PrettyPrinter(indent=4)

transcript = transcript(AUDIO_FILENAME="audio.wav", num_speakers=2)
pp.pprint(transcript)