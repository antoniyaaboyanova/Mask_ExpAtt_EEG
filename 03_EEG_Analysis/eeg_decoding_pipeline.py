### Libs & PATHS
import argparse
from MaskExp.EEG import decode_neutral_category, decode_exp_unexp_category, decode_exp_unexp, decode_neutral, decode_exp_unexp_unMatch

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, required=True, help='Subject ID')
args = parser.parse_args()
sub = args.sub

### Temporal Generalization decoding
#print(f"<<< EEG Neutral Decoding - cue baseline, Subject {sub}>>>")
#decode_neutral(sub, epoched_ver="cue", imgPerm=1, testsize=0.2, group_size=4, whitening=False, TempGen = True, model="lda")

#print(f"<<< EEG Exp, Unexp Decoding - cue baseline, Subject {sub}>>>")
#decode_exp_unexp(sub, epoched_ver="cue", imgPerm=20, group_size=4, whitening=False, TempGen=True, model="lda")

#print(f"<<< EEG Exp, Unexp Decoding - cue baseline, Subject {sub}>>>")
#decode_exp_unexp_unMatch(sub, epoched_ver="cue", group_size=4, whitening=False, TempGen=False, model='lda')

print(f"<<< EEG Category decoding - cue baseline, Subject {sub}>>>")
decode_neutral_category(sub)
decode_exp_unexp_category(sub)

#print("<<< EEG searchlight >>>>")
#decode_searchlight_exp_unexp(sub, epoched_ver="cue", imgPerm=10, group_size=4,
                                  #whitening=False, TempGen=False, model='lda')
#print(f"<<< EEG Decoding - target baseline, Subject {sub}>>>")
#decode_general(sub, epoched_ver="target", imgPerm=10, testsize=0.2, group_size=4, whitening=False)




