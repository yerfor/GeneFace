import soundfile as sf
import numpy as np
from argparse import ArgumentParser
from data_gen.process_lrs3.process_audio_hubert import get_hubert_from_16k_speech

parser = ArgumentParser()
parser.add_argument('--video_id', type=str, default='Yixing', help='')
args = parser.parse_args()

person_id = args.video_id
wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
hubert_npy_name = f"data/processed/videos/{person_id}/hubert.npy"
speech_16k, _ = sf.read(wav_16k_name)
hubert_hidden = get_hubert_from_16k_speech(speech_16k)
np.save(hubert_npy_name, hubert_hidden.detach().numpy())
print(f"Hubert extracted at {hubert_npy_name}")