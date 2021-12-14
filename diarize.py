import os
import wget
import json
from omegaconf import OmegaConf

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels


def read_file(path_to_file):
    with open(path_to_file) as f:
        contents = f.read().splitlines()
    return contents


def transcript(AUDIO_FILENAME, num_speakers, directory="diarize", extended=False):
    AUDIO_BASENAME = AUDIO_FILENAME[:AUDIO_FILENAME.rfind('.')]
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


    cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')

    pretrained_speaker_model='ecapa_tdnn'
    cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath

    # Directory to store intermediate files and prediction outputs
    cfg.diarizer.out_dir = data_dir 
    cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
    cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
    cfg.diarizer.clustering.parameters.oracle_num_speakers=True

    # USE VAD generated from ASR timestamps
    cfg.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.asr.parameters.asr_based_vad = True
    cfg.diarizer.asr.parameters.threshold=300


    asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer.asr.parameters)
    asr_diar_offline.root_path = cfg.diarizer.out_dir

    AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)
    asr_diar_offline.AUDIO_RTTM_MAP = AUDIO_RTTM_MAP
    asr_model = asr_diar_offline.set_asr_model(cfg.diarizer.asr.model_path)


    # Generating words and timestamps from audio
    word_list, word_ts_list = asr_diar_offline.run_ASR(asr_model)

    # Creates .rttm file
    score = asr_diar_offline.run_diarization(cfg, word_ts_list)

    predicted_speaker_label_rttm_path = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.rttm"
    pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)

    # Matches diarization ouput and word_list; creates transcript .txt file
    asr_output_dict = asr_diar_offline.write_json_and_transcript(word_list, word_ts_list)

    transcription_path_to_file = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.txt"
    transcript = read_file(transcription_path_to_file)


    # Extended data of transcript
    if(extended):
        transcription_path_to_file = f"{data_dir}/pred_rttms/{AUDIO_BASENAME}.json"
        json_contents = read_file(transcription_path_to_file)
        return json_contents

    return transcript