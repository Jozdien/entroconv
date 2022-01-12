import os
import diarize
import audio_utils
import transcript_utils

AUDIO_FILENAME = "../samples/audio_short.wav"
num_speakers = 2
segment_lengths = 30
sentence_length = 10


def annotate_audio(AUDIO_FILENAME, num_speakers, segment_lengths, sentence_length):
	complete_transcript = []

	# Converting other audio formats to wav
	if audio_utils.get_extension(AUDIO_FILENAME) != "wav":
		AUDIO_FILENAME = audio_utils.audio_convert(AUDIO_FILENAME)

	# Converting from stereo to mono and sampling rate to 16000
	audio_utils.format_audio(AUDIO_FILENAME)

	# Reading audio file as AudioSegment object
	audio = audio_utils.audio_segment(AUDIO_FILENAME)

	# Defining parameters for diarization model now to prevent repeating every turn
	params = diarize.config()

	# Timestamps of every word in the audiio file
	words, word_ts = diarize.timestamps(AUDIO_FILENAME, num_speakers=num_speakers)
	timestamps = word_ts[0]

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
		transcript_formatted = transcript_utils.transcript_sets(transcript=transcript_raw)
		# Converts formatted transcript into annotated form
		transcript_annotated = transcript_utils.transcript_entropies(transcript=transcript_formatted)
		# Removes last speaker section if smaller than one sentence, redefines beginning timestamp of next segment to hold cut section
		start = transcript_utils.remove_tail(transcript_annotated=transcript_annotated, transcript_extended=transcript_extended, sen_len=sentence_length, next_start=next_start, start=start)
		# Splits annotated transcript into sentences.
		sentences_annotated = transcript_utils.transcript_sentences(transcript=transcript_annotated, sen_len=sentence_length)

		# Deleting temporary file
		os.remove(SPLIT_FILENAME)

		complete_transcript.append(sentences_annotated)
		i += 1

	return complete_transcript


if __name__ == '__main__':

	transcript = annotate_audio(AUDIO_FILENAME=AUDIO_FILENAME, num_speakers=num_speakers, segment_lengths=segment_lengths, sentence_length=sentence_length)

	print(transcript)