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

### NLP Stuff

```shell
pip install nltk
```

### Miscellaneous audio processing

```shell
pip install pydub
pip install fleep
```

## Error Fixes

If you see this error: `OSError: libtorch_hip.so: cannot open shared object file: No such file or directory`, run the following command:

```shell
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```