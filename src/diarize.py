import os
import wget
import json
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels


def read_file(path_to_file):
    with open(path_to_file) as f:
        contents = f.read().splitlines()
    return contents


def config(AUDIO_FILENAME, num_speakers, directory="diarize"):
    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT, directory)

    # Unless the directory is left blank and files are stored in the root, make new directory
    if directory:
        os.makedirs(data_dir, exist_ok=True)


    # Pretrained ASR model parameters
    CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization_with_asr.yaml"

    if not os.path.exists(os.path.join(data_dir,'offline_diarization_with_asr.yaml')):
        CONFIG = wget.download(CONFIG_URL, data_dir)
    else:
        CONFIG = os.path.join(data_dir,'offline_diarization_with_asr.yaml')

    cfg = OmegaConf.load(CONFIG)

    meta = {
        'audio_filepath': AUDIO_FILENAME, 
        'offset': 0, 
        'duration': None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': num_speakers, 
        'rttm_filepath': None, 
        'uem_filepath' : None
    }

    with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    pretrained_speaker_model='ecapa_tdnn'

    # Directory to store intermediate files and prediction outputs
    cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')
    cfg.diarizer.out_dir = data_dir 
    cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
    cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
    cfg.diarizer.clustering.parameters.oracle_num_speakers=True

    # USE VAD generated from ASR timestamps
    cfg.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.asr.parameters.asr_based_vad = True
    cfg.diarizer.asr.parameters.threshold=100 # Original value => 300
    cfg.diarizer.asr.parameters.decoder_delay_in_sec=0.2

    asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
    asr_model = asr_ts_decoder.set_asr_model()

    # Generating words and timestamps from audio
    word_list, word_ts_list = asr_ts_decoder.run_ASR(asr_model)

    # asr_diar_offline.root_path = cfg.diarizer.out_dir


    params = {  
        "cfg": cfg, 
        "data_dir": data_dir
    }


    AUDIO_BASENAME = AUDIO_FILENAME[AUDIO_FILENAME.rfind('/')+1:AUDIO_FILENAME.rfind('.')]

    return params, word_list[AUDIO_BASENAME], word_ts_list[AUDIO_BASENAME]


def transcript(AUDIO_FILENAME, num_speakers, params=None, directory="diarize"):
    if not params:
        params, _, _ = config(AUDIO_FILENAME=AUDIO_FILENAME, num_speakers=num_speakers, directory=directory)

    data_dir = params["data_dir"]
    cfg = params["cfg"]

    AUDIO_BASENAME = AUDIO_FILENAME[AUDIO_FILENAME.rfind('/')+1:AUDIO_FILENAME.rfind('.')]


    meta = {
        'audio_filepath': AUDIO_FILENAME, 
        'offset': 0, 
        'duration': None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': num_speakers, 
        'rttm_filepath': None, 
        'uem_filepath' : None
    }

    with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')


    cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')

    asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
    asr_model = asr_ts_decoder.set_asr_model()

    # Generating words and timestamps from audio
    word_list, word_ts_list = asr_ts_decoder.run_ASR(asr_model)

    asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset

    # AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)
    # asr_diar_offline.AUDIO_RTTM_MAP = AUDIO_RTTM_MAP
    # asr_model = asr_diar_offline.set_asr_model(cfg.diarizer.asr.model_path)

    # Generating words and timestamps from audio
    # word_list, word_ts_list = asr_diar_offline.run_ASR(asr_model)

    # Creates .rttm file
    diar, score = asr_diar_offline.run_diarization(cfg, word_ts_list)

    predicted_speaker_label_rttm_path = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.rttm"
    pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)

    # Matches diarization ouput and word_list; creates transcript .txt file
    asr_diar_offline.get_transcript_with_speaker_labels(diar, word_list, word_ts_list)

    # Matches diarization ouput and word_list; creates transcript .txt file
    # asr_output_dict = asr_diar_offline.write_json_and_transcript(word_list, word_ts_list)

    transcription_path_to_file = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.txt"
    transcript = read_file(transcription_path_to_file)


    # Extended data of transcript
    transcription_path_to_file = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.json"
    with open(transcription_path_to_file) as f:
        transcript_extended = json.load(f)

    return transcript, transcript_extended