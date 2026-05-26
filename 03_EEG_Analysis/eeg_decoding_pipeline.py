### Libs & PATHS
import argparse
from MaskExp.EEG import *

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, required=True, help='Subject ID')
args = parser.parse_args()
sub = args.sub

### Preprocessing
print(f"<<< EEG Preprocessing - cue baseline, Subject {sub}>>>")
data_val = preprocess_eeg_cue(sub=sub, lowpass=40, resample=100)

print(f"<<< EEG Preprocessing - target baseline, Subject {sub}>>>")
preprocess_eeg(sub=sub, lowpass=40, resample=100)

### Temporal Generalization decoding
print(f"<<< EEG Decoding - cue baseline, Subject {sub}>>>")
decode_general(sub, epoched_ver="cue", imgPerm=10, testsize=0.2, group_size=4, whitening=False)

print(f"<<< EEG Decoding - target baseline, Subject {sub}>>>")
decode_general(sub, epoched_ver="target", imgPerm=10, testsize=0.2, group_size=4, whitening=False)

### Cross-decoding mask level 

### Expectation decoding 


