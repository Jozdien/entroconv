# Conversational Entropy

## Requirements

In order to run `program.py`, install all the requirements below.

### HuggingFace Transformers + PyTorch

```shell
pip install transformers[torch]
```

### Speaker Diarization

#### Dependencies

```shell
pip install wget
apt-get install sox libsndfile1 ffmpeg
pip install unidecode
```

#### NeMo

```shell
python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

#### TorchAudio

```shell
pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### Miscellaneous audio processing

```shell
pip install pydub
pip install fleep
```