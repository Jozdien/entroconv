import nemo.collections.asr as nemo_asr
import numpy as np
from IPython.display import Audio, display
import librosa
import os
import wget
import matplotlib.pyplot as plt

import nemo
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE

import glob

import pprint
pp = pprint.PrettyPrinter(indent=4)


ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
if not os.path.exists(os.path.join(data_dir,'an4_diarize_test.wav')):
    AUDIO_FILENAME = wget.download(an4_audio_url, data_dir)
else:
    AUDIO_FILENAME = os.path.join(data_dir,'an4_diarize_test.wav')

audio_file_list = glob.glob(f"{data_dir}/*.wav")
print("Input audio file list: \n", audio_file_list)

signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)
display(Audio(signal,rate=sample_rate))


def display_waveform(signal,text='Audio',overlay_color=[]):
    fig,ax = plt.subplots(1,1)
    fig.set_figwidth(20)
    fig.set_figheight(2)
    plt.scatter(np.arange(len(signal)),signal,s=1,marker='o',c='k')
    if len(overlay_color):
        plt.scatter(np.arange(len(signal)),signal,s=1,marker='o',c=overlay_color)
    fig.suptitle(text, fontsize=16)
    plt.xlabel('time (secs)', fontsize=18)
    plt.ylabel('signal strength', fontsize=14);
    plt.axis([0,len(signal),-0.5,+0.5])
    time_axis,_ = plt.xticks();
    plt.xticks(time_axis[:-1],time_axis[:-1]/sample_rate);
    
COLORS="b g c m y".split()

def get_color(signal,speech_labels,sample_rate=16000):
    c=np.array(['k']*len(signal))
    for time_stamp in speech_labels:
        start,end,label=time_stamp.split()
        start,end = int(float(start)*16000),int(float(end)*16000),
        if label == "speech":
            code = 'red'
        else:
            code = COLORS[int(label.split('_')[-1])]
        c[start:end]=code
    
    return c 


display_waveform(signal)


from omegaconf import OmegaConf
import shutil
CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization_with_asr.yaml"

if not os.path.exists(os.path.join(data_dir,'offline_diarization_with_asr.yaml')):
    CONFIG = wget.download(CONFIG_URL, data_dir)
else:
    CONFIG = os.path.join(data_dir,'offline_diarization_with_asr.yaml')
   
cfg = OmegaConf.load(CONFIG)
print(OmegaConf.to_yaml(cfg))


# Create a manifest for input with below format. 
# {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-", 
# "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}
import json
meta = {
    'audio_filepath': AUDIO_FILENAME, 
    'offset': 0, 
    'duration':None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': 2, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}
with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
    json.dump(meta,fp)
    fp.write('\n')


cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')
print(cfg.diarizer.manifest_filepath)


pretrained_speaker_model='ecapa_tdnn'
cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
cfg.diarizer.out_dir = data_dir #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
cfg.diarizer.clustering.parameters.oracle_num_speakers=True

# USE VAD generated from ASR timestamps
cfg.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
cfg.diarizer.oracle_vad = False # ----> ORACLE VAD 
cfg.diarizer.asr.parameters.asr_based_vad = True
cfg.diarizer.asr.parameters.threshold=300


from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer.asr.parameters)
asr_diar_offline.root_path = cfg.diarizer.out_dir

AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)
asr_diar_offline.AUDIO_RTTM_MAP = AUDIO_RTTM_MAP
asr_model = asr_diar_offline.set_asr_model(cfg.diarizer.asr.model_path)


word_list, word_ts_list = asr_diar_offline.run_ASR(asr_model)

print("Decoded word output: \n", word_list[0])
print("Word-level timestamps \n", word_ts_list[0])


score = asr_diar_offline.run_diarization(cfg, word_ts_list)


def read_file(path_to_file):
    with open(path_to_file) as f:
        contents = f.read().splitlines()
    return contents


predicted_speaker_label_rttm_path = f"{data_dir}/pred_rttms/an4_diarize_test.rttm"
pred_rttm = read_file(predicted_speaker_label_rttm_path)

pp.pprint(pred_rttm)


from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)

color = get_color(signal, pred_labels)
display_waveform(signal,'Audio with Speaker Labels', color)
display(Audio(signal,rate=16000))


asr_output_dict = asr_diar_offline.write_json_and_transcript(word_list, word_ts_list)


transcription_path_to_file = f"{data_dir}/pred_rttms/an4_diarize_test.txt"
transcript = read_file(transcription_path_to_file)
pp.pprint(transcript)


# Extended data of transcript
'''
transcription_path_to_file = f"{data_dir}/pred_rttms/an4_diarize_test.json"
json_contents = read_file(transcription_path_to_file)
pp.pprint(json_contents)
'''