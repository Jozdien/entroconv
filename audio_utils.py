from pydub import AudioSegment
import fleep
import math
import sys


def audio_segment(AUDIO_FILENAME):
	return AudioSegment.from_file(AUDIO_FILENAME, format="wav")


def get_extension(AUDIO_FILENAME):
	with open(AUDIO_FILENAME, "rb") as file:
		info = fleep.get(file.read(128))

	if info.type != ['audio']:
		sys.exit(f"File format: \"{info.type[0]}\".  Use an audio file.")

	return info.extension[0]


def get_duration(AUDIO_FILENAME):
	audio = AudioSegment.from_file(AUDIO_FILENAME, format="wav")

	return audio.duration_seconds


def audio_convert(AUDIO_FILENAME):
	extension = get_extension(AUDIO_FILENAME)
	CONVERTED_FILENAME = AUDIO_FILENAME[:-len(extension)] + "wav"

	audio = AudioSegment.from_file(AUDIO_FILENAME, format=extension)

	audio.export(CONVERTED_FILENAME, format="wav")

	return CONVERTED_FILENAME


def single_split(audio, SPLIT_FILENAME, start, seg_len, timestamps):
	timestamps_begin = [item[0] for item in timestamps]
	timestamps_end = [item[1] for item in timestamps]

	end = min(timestamps_end, key=lambda x:abs(x - (start + seg_len)))

	if start == end:
		return -1

	start_split = start * 1000 - 50
	end_split = end * 1000 + 50

	new_audio = audio[start_split:end_split]

	new_audio.export(SPLIT_FILENAME, format="wav")

	# Set start to starting time of next word after end, or to -1 if reached last word
	try:
		start = timestamps_begin[next(x[0] for x in enumerate(timestamps_begin) if x[1] > end)]
	except:
		start = -1

	return start


def split(AUDIO_FILENAME, seg_len, timestamps):
	audio = AudioSegment.from_file(AUDIO_FILENAME, format="wav")
	new_files = []

	start = timestamps[0][0]

	i = 0
	while start >= 0:
		SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"

		start = single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, seg_len=seg_len, timestamps=timestamps)

		if start >= 0:
			new_files.append(SPLIT_FILENAME)
			
		i += 1

	return new_files