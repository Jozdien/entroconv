import os
import text_utils

TRANSCRIPT_FILENAME = "../samples/test_transcript.txt"
ESSAY_FILENAME = "../samples/test_essay.txt"
AUDIO_FILENAME = "../samples/audio_vshort.wav"
num_speakers = 2
segment_lengths = 30
sentence_length = 10


def annotate_audio(AUDIO_FILENAME, num_speakers, segment_lengths, sentence_length):
	import diarize
	import audio_utils
	
	complete_transcript = []

	# Converting other audio formats to wav
	if audio_utils.get_extension(AUDIO_FILENAME) != "wav":
		AUDIO_FILENAME = audio_utils.audio_convert(AUDIO_FILENAME)

	# Converting from stereo to mono and sampling rate to 16000
	audio_utils.format_audio(AUDIO_FILENAME)

	# Reading audio file as AudioSegment object
	audio = audio_utils.audio_segment(AUDIO_FILENAME=AUDIO_FILENAME)

	# Defining parameters for diarization model now to prevent repeating every turn; Timestamps of every word in the audiio file
	params, words, word_ts = diarize.config(AUDIO_FILENAME=AUDIO_FILENAME, num_speakers=num_speakers)
	timestamps = word_ts

	# Timestamp of first word in the audio file
	start = timestamps[0][0]

	i = 1
	while start >= 0:
		# Temporary filename for this segment
		SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"

		# Creating audio split segment, storing timestamp of when next segment begins
		next_start = audio_utils.single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, seg_len=segment_lengths, timestamps=timestamps)

		# Transcript of audio segment, both in reduced form and complete form
		transcript_raw, transcript_extended = diarize.transcript(AUDIO_FILENAME=SPLIT_FILENAME, num_speakers=num_speakers, params=params)

		# Converts transcript into name, text pairs
		transcript_formatted = text_utils.audio_transcript_to_sets(transcript=transcript_raw)
		# Converts formatted transcript into annotated form
		transcript_annotated = text_utils.get_transcript_entropies_words(transcript=transcript_formatted)
		# Removes last speaker section if smaller than one sentence, redefines beginning timestamp of next segment to hold cut section
		start = text_utils.remove_tail(transcript_annotated=transcript_annotated, transcript_extended=transcript_extended, sen_len=sentence_length, next_start=next_start, start=start)
		# Splits annotated transcript into sentences.
		sentences_annotated = text_utils.transcript_to_sentences_arbitrary(transcript=transcript_annotated, sen_len=sentence_length)

		# Deleting temporary file
		os.remove(SPLIT_FILENAME)

		complete_transcript.append(sentences_annotated)
		i += 1

	return complete_transcript


def annotate_transcript(FILENAME):
	"""
	Assuming transcript is stored in the file in the following format:
		<speaker_name>: <speaker_text>\n<speaker_name>: <speaker_text>\n...

	"""

	with open(FILENAME) as f:
		transcript = f.read().splitlines()

	# Formatting raw text into (speaker, text) pairs
	transcript = text_utils.transcript_to_speakers(transcript)

	# Splitting text into sentences
	speaker_sentences = text_utils.transcript_to_sentences_proper(transcript)

	# Getting entropies of sentences
	sentence_entropies = text_utils.get_transcript_entropies_sentences(speaker_sentences)

	return sentence_entropies


def annotate_essay(FILENAME):
	annotated = []

	with open(FILENAME) as f:
		essay = f.read()

	essay_sentences = text_utils.essay_to_sentences(essay)

	sentence_entropies = text_utils.get_essay_entropies(essay_sentences)

	return sentence_entropies


if __name__ == '__main__':

	annotated_audio_transcript = annotate_audio(AUDIO_FILENAME=AUDIO_FILENAME, num_speakers=num_speakers, segment_lengths=segment_lengths, sentence_length=sentence_length)
	print(annotated_audio_transcript)
	
	# annotated_transcript = annotate_transcript(FILENAME=TRANSCRIPT_FILENAME)
	# print(annotated_transcript)
	
	# annotated_essay = annotate_essay(FILENAME=ESSAY_FILENAME)
	# print(annotated_essay)