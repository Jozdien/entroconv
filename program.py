from diarize import transcript
from gpt2 import word_entropy
from audio_utils import audio_convert, split_audio, get_extension, get_duration

AUDIO_FILENAME = "audio.wav"
num_speakers = 2
segment_lengths = 30


# Converts raw transcript to list of sets of the form (speaker_name, speaker_text)
# speaker_text is the list of words spoken by speaker_name in a line
def transcript_sets(transcript):
	ret = []

	for line in transcript:
		name_start = line.index("]") + 2
		name_end = line.index(":", line.find("speaker"))

		text_start = name_end + 2

		name = line[name_start:name_end]
		text = line[text_start:].split()

		line_set = (name, text)

		ret.append(line_set)

	return ret


# Converts "list of sets" transcript to a list of sets of the form (speaker_name, annotated_speaker_text)
# annotated_speaker_text is a list of sets of the form (word, word_entropy)
def transcript_entropies(transcript):
	ret = []

	# Keeps track of the entire transcript so far to be used as prefix to new word
	running_prefix = ""

	for line_set in transcript:
		name, text = line_set
		running_prefix += name + ":"

		ret_set = (name, [])

		for word in text:
			word_formatted = " " + word
			word_surprise = word_entropy(running_prefix, word_formatted)

			word_set = (word, word_surprise)

			ret_set[1].append(word_set)

			running_prefix += word_formatted

		ret.append(ret_set)
		running_prefix += "\n"

	return ret



if __name__ == '__main__':

	if get_extension(AUDIO_FILENAME) != "wav":
		AUDIO_FILENAME = audio_convert(AUDIO_FILENAME)

	AUDIO_FILENAME_LIST = split_audio(AUDIO_FILENAME, seg_len=segment_lengths)

	with open("annotated_transcript.txt", "w+") as file:
		for AUDIO_FILENAME in AUDIO_FILENAME_LIST:
			transcript = transcript(AUDIO_FILENAME=AUDIO_FILENAME, num_speakers=num_speakers)

			transcript_formatted = transcript_sets(transcript)
			transcript_annotated = transcript_entropies(transcript_formatted)

			for (name, text) in transcript_annotated:
				file.write(f"{name}: {text}")
				file.write("\n")