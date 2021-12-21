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


def single_split(audio, SPLIT_FILENAME, start, end):
	if end == 0:
		new_audio = audio[start:]
	else:
		new_audio = audio[start:end]

	new_audio.export(SPLIT_FILENAME, format="wav")


def split_audio(AUDIO_FILENAME, seg_len=1):
	audio = AudioSegment.from_file(AUDIO_FILENAME, format="wav")
	new_files = []

	file_prefix = AUDIO_FILENAME[:-4]

	duration = math.ceil(audio.duration_seconds)

	for i, val in enumerate(range(0, duration, seg_len)):
		SPLIT_FILENAME = f"{AUDIO_FILENAME}-{i:05}.wav"
		start = seg_len * i
		end = seg_len * (i + 1)

		if end > duration:
			end = 0

		single_split(audio=audio, SPLIT_FILENAME=SPLIT_FILENAME, start=start, end=end)

		new_files.append(SPLIT_FILENAME)

	return new_files