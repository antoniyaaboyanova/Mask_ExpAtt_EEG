### Libs & PATHS
import argparse
from MaskExp.EEG import decode_general, decode_exp_unexp

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, required=True, help='Subject ID')
args = parser.parse_args()
sub = args.sub

### Temporal Generalization decoding
#print(f"<<< EEG Decoding - cue baseline, Subject {sub}>>>")
#decode_general(sub, epoched_ver="cue", imgPerm=10, testsize=0.2, group_size=4, whitening=False, TempGen = True, model="lda")


print(f"<<< EEG Decoding - cue baseline, exp, unexp Subject {sub}>>>")
decode_exp_unexp(sub, epoched_ver="cue", imgPerm=100, group_size=4, whitening=False, TempGen =False, model="lda")

#print(f"<<< EEG Decoding - target baseline, Subject {sub}>>>")
#decode_general(sub, epoched_ver="target", imgPerm=10, testsize=0.2, group_size=4, whitening=False)

### Cross-decoding mask level 

### Expectation decoding 


