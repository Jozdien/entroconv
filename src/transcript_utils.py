import gpt2

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


def remove_tail(transcript_annotated, transcript_extended, sen_len, next_start, start):
	"""
	Removes last speaker section if smaller than one sentence and redefines beginning timestamp of next segment to hold cut section

	Args:
		transcript_annotated: list of sets of the form (speaker_name, annotated_text), where annotated_text is a list of sets of the form (word, entropy)

		transcript_extended: dict; "words" tag contains list of dicts containing text, start_time, end_time, and speaker_label of every word

		sen_len: number of words in a sentence

		next_start: current beginning timestamp of next split segment

		start: beginning timestamp of current split segment

	Returns:
		Beginning timestamp of next split segment

	"""

	if len(transcript_annotated[-1][1]) < sen_len and next_start >= 0:
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

	return start


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