import nltk
import gpt2

def audio_transcript_to_sets(transcript):
	"""
	Formats raw audio transcript into name, text pairs.

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


def transcript_to_speakers(transcript):
	"""
	Splits transcript lines into list of (speaker, speech) sets

	Args:
		transcript: list of strings containing one speaker and their speech

	Returns:
		A list of sets of the form (speaker_name, speech)
		
	"""

	transcript = [(item[:item.find(':')], item[item.find(':')+2:]) for item in transcript]

	return transcript


def get_transcript_entropies_words(transcript):
	"""
	Converts formatted transcript into annotated transcript.

	Args:
		transcript: list of sets of the form(speaker_name, text) where text is a list of words

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
			word_surprise = gpt2.entropy_of_next(running_prefix, word_formatted)

			word_set = (word, word_surprise)

			ret_set[1].append(word_set)

			running_prefix += word_formatted

		ret.append(ret_set)
		running_prefix += "\n"

	return ret


def get_transcript_entropies_sentences(transcript):
	"""
	Gets entropies for sentences in transcript

	Args:
		transcript: list of sets of the form(speaker_name, sentences) where sentences is a list of strings

	Returns:
		A list of sets of the form (speaker_name, annotated_sentences), where annotated_sentences is a list of sets of the form (sentence, avg_entropy)

	"""

	ret = []

	# Keeps track of the entire transcript so far to be used as prefix to new word
	running_prefix = ""

	for (speaker, sentence_list) in transcript:
		running_prefix += speaker + ":"

		ret_set = (speaker, [])

		for sentence in sentence_list:
			sentence_formatted = " " + sentence
			surprise = gpt2.entropy_of_next(running_prefix, sentence_formatted)

			sentence_set = (sentence, surprise)

			ret_set[1].append(sentence_set)

			running_prefix += sentence_formatted

		ret.append(ret_set)
		running_prefix += "\n"

	return ret


def get_essay_entropies(essay_sentences):
	"""
	Gets entropies of sentences in essay

	Args:
		essay_sentences: list of strings, each containing a sentence

	Returns:
		A list of sets of the form (setence, avg_entropy)

	"""

	ret = []

	essay_sentences = [essay_sentences[0]] + [" " + sentence for sentence in essay_sentences[1:]]

	# Keeps track of the entire transcript so far to be used as prefix to new word
	running_prefix = ""

	for sentence in essay_sentences:
		surprise = gpt2.entropy_of_next(running_prefix, sentence)

		ret.append((sentence, surprise))

		running_prefix += sentence

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


def transcript_to_sentences_arbitrary(transcript, sen_len):
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


def transcript_to_sentences_proper(transcript):
	"""
	Splits transcript into sentences.

	Args:
		transcript: list of sets of the form (speaker_name, text), where text is a string

	Returns:
		A list of sets of the form (speaker_name, sentences), where sentences is a list of strings

	"""

	try:
		nltk.data.find('tokenizers/punkt')
	except:
		nltk.download('punkt')
	ret = []

	for (speaker, text) in transcript:
		sentences = nltk.tokenize.sent_tokenize(text)

		ret.append((speaker, sentences))

	return ret


def essay_to_sentences(essay):
	"""
	Splits essay into sentences.

	Args:
		essay: string of raw data from essay file

	Returns:
		A list of strings, each containing a sentence

	"""

	try:
		nltk.data.find('tokenizers/punkt')
	except:
		nltk.download('punkt')
	
	return nltk.tokenize.sent_tokenize(essay)