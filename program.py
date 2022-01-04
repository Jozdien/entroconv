import diarize
import gpt2
import audio_utils

AUDIO_FILENAME = "audio.wav"
num_speakers = 2
segment_lengths = 150
sentence_length = 10


def transcript_sets(transcript):
	"""
	Formats raw transcript into name, text pairs.

	Args:
		transcript: list of strings of the form <timestamp> <speaker_name> <text>

	Returns:
		A list of sets of the form (speaker_name, text)
		
	"""

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


def transcript_entropies(transcript):
	"""
	Converts formatted transcript into annotated transcript.

	Args:
		transcript: list of sets of the form(speaker_name, text)

	Returns:
		A list of sets of the form (speaker_name, annotated_text), where annotated_text is a list of sets of the form (word, entropy)

	"""

	ret = []

	# Keeps track of the entire transcript so far to be used as prefix to new word
	running_prefix = ""

	for line_set in transcript:
		name, text = line_set
		running_prefix += name + ":"

		ret_set = (name, [])

		for word in text:
			word_formatted = " " + word
			word_surprise = gpt2.word_entropy(running_prefix, word_formatted)

			word_set = (word, word_surprise)

			ret_set[1].append(word_set)

			running_prefix += word_formatted

		ret.append(ret_set)
		running_prefix += "\n"

	return ret


def chunks(lst, n):
	"""
	Yield successive n-sized chunks from lst
	"""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def transcript_sentences(transcript, sen_len):
	"""
	Splits annotated transcript into sentences.

	Args:
		transcript: list of sets of the form (speaker_name, annotated_text), where annotated_text is a list of sets of the form (word, entropy)

		sen_len: number of words in a sentence

	Returns:
		A list of sets of the form (speaker_name, annotated_sentences), where annotated_sentences is a list of sets of the form (sentence, avg_entropy)

	"""

	ret = []

	for (speaker, annotated_text) in transcript:
		ret_set = (speaker, [])

		sentence_chunks = list(chunks(annotated_text, sen_len))

		if len(sentence_chunks[-1]) in [1, 2] and len(sentence_chunks) > 1:
			for word in sentence_chunks[-1]:
				sentence_chunks[-2].append(word)
			del sentence_chunks[-1]

		for sentence in sentence_chunks:
			words = [word[0] for word in sentence]
			word_ents = [word[1] for word in sentence]

			word_str = " ".join(words)
			entropy = sum(word_ents) / len(word_ents)

			ret_set[1].append((word_str, entropy))

		ret.append(ret_set)

	return ret




if __name__ == '__main__':

	if audio_utils.get_extension(AUDIO_FILENAME) != "wav":
		AUDIO_FILENAME = audio_utils.audio_convert(AUDIO_FILENAME)

	# Converting to mono and sampling rate to 16000
	audio_utils.format_audio(AUDIO_FILENAME)

	audio = audio_utils.audio_segment(AUDIO_FILENAME)
	params = diarize.config()

	words, word_ts = diarize.timestamps(AUDIO_FILENAME, num_speakers=num_speakers)
	timestamps = word_ts[0]

	with open("annotated_transcript.txt", "w+") as file:
		start = timestamps[0][0]

		i = 1
		while start >= 0:
			SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"

			next_start = audio_utils.single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, seg_len=segment_lengths, timestamps=timestamps)

			transcript_raw, transcript_extended = diarize.transcript(AUDIO_FILENAME=SPLIT_FILENAME, num_speakers=num_speakers, params=params)

			transcript_formatted = transcript_sets(transcript=transcript_raw)
			transcript_annotated = transcript_entropies(transcript=transcript_formatted)

			if len(transcript_annotated[-1][1]) < sentence_length and next_start >= 0:
				speaker = transcript_annotated[-1][0]
				del transcript_annotated[-1]

				for word in reversed(transcript_extended["words"]):
					if word["speaker_label"] == speaker:
						start_temp = start + word["start_time"]
					else:
						break

				start = start_temp
			else:
				start = next_start


			sentences_annotated = transcript_sentences(transcript=transcript_annotated, sen_len=sentence_length)

			file.write(f"Segment {i}\n-----------------\n")
			for name, text in sentences_annotated:
				file.write(f"{name}: {text}")
				file.write("\n")
			file.write("\n")

			i += 1