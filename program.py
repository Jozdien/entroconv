from diarize import config, timestamps, transcript
from gpt2 import word_entropy
from audio_utils import audio_segment, audio_convert, single_split, get_extension

AUDIO_FILENAME = "audio_long.wav"
num_speakers = 2
segment_lengths = 30
sentence_length = 10


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

	audio = audio_segment(AUDIO_FILENAME)
	params = config()

	words, word_ts = timestamps(AUDIO_FILENAME, num_speakers=num_speakers)
	timestamps = word_ts[0]

	print(words)
	print(word_ts)

	with open("annotated_transcript.txt", "w+") as file:
		start = timestamps[0][0]

		i = 1
		while start >= 0:
			SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"

			start = single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, seg_len=segment_lengths, timestamps=timestamps)

			transcript_raw, transcript_extended = transcript(AUDIO_FILENAME=SPLIT_FILENAME, num_speakers=num_speakers, params=params)

			transcript_formatted = transcript_sets(transcript_raw)
			transcript_annotated = transcript_entropies(transcript_formatted)

			# for item in transcript_annotated:
			# 	print(item)

			# print("------------")

			if len(transcript_annotated[-1][1]) < sentence_length and start >= 0:
				speaker = transcript_annotated[-1][0]
				del transcript_annotated[-1]

				# for item in transcript_annotated:
				# 	print(item)

				for word in reversed(transcript_extended["words"]):
					if word["speaker_label"] == speaker:
						start_temp = start + word["start_time"]
					else:
						break

				start = start_temp

			print(start)

			file.write(f"Segment {i}\n-----------------\n")
			for name, text in transcript_annotated:
				file.write(f"{name}: {text}")
				file.write("\n")
			file.write("\n")

			i += 1