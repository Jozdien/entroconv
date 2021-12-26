from pydub import AudioSegment
import fleep
import math
import sys


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


def single_split(audio, SPLIT_FILENAME, start, seg_len, timestamps_end):
	end = min(timestamps_end, key=lambda x:abs(x - (start + seg_len)))

	if start == end:
		return -1

	new_audio = audio[start*1000:end*1000]

	new_audio.export(SPLIT_FILENAME, format="wav")

	return end


def split(AUDIO_FILENAME, seg_len, timestamps):
	audio = AudioSegment.from_file(AUDIO_FILENAME, format="wav")
	new_files = []

	file_prefix = AUDIO_FILENAME[:-4]

	start = timestamps[0][0]

	timestamps_end = [item[1] for item in timestamps]

	i = 0
	while start >= 0:
		SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"

		start = single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, seg_len=seg_len, timestamps_end=timestamps_end)

		if start >= 0:
			new_files.append(SPLIT_FILENAME)
			
		i += 1

	return new_files