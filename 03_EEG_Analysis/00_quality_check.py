import mne
import numpy as np
import pandas as pd

from MaskExp.data_helpers import *
from MaskExp.EEG import preprocess_eeg, preprocess_eeg_cue
import argparse

# parse
parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, required=True, help='Subject ID')
args = parser.parse_args()
sub = args.sub

PATH = "/projects/archiv/DataStore_Boyanova/Mask_ExpAtt_EEG/"
# check trig timings and id numbers

# I don't want to see all the output
mne.set_log_level('error')

# load raw files
folder_name = f"sub-{sub:04d}"
sub_folder = os.path.join(PATH, "eeg_raw", folder_name)
files = os.listdir(sub_folder)

vhdrs = [x for x in files if ".vhdr" in x]
raw = []
allowed_events = get_main_events()
rename_dict = condition_events(allowed_events=allowed_events,  rename_cond="stims")

for vhdr in vhdrs:
    vhdr_file = os.path.join(sub_folder, vhdr)
    raw.append(mne.io.read_raw_brainvision(vhdr_file, preload=True))

raw = mne.concatenate_raws(raw)
# make sure eye channels are marked correctly 
raw.set_channel_types({
    'VEOG1':'eog',
    'VEOG2':'eog',
    'HEOG1':'eog',
    'HEOG2':'eog'
})


raw, _ = mne.set_eeg_reference(raw, ref_channels='average', ch_type = 'eeg')

events, events_ids = mne.events_from_annotations(raw)
condition_keys = list(set(rename_dict.values()))

##### Check event timings #####
sfreq = raw.info['sfreq']
target_codes = [events_ids[name] for name in condition_keys if name in events_ids]
event_samples = events[:, 0]
event_ids = events[:, 2]

## 1. cue to image
is_target = np.isin(event_ids, target_codes)
start_indices = np.where(event_ids == 1)[0]

intervals = {"start_sample":[],
            "target_sample":[],
            "target_id": [],
            "dt_sec": []}

for start_idx in start_indices:
    next_target_idx = np.where(is_target & (np.arange(len(events)) > start_idx))[0]
    if next_target_idx.size == 0:
        continue
    next_idx = next_target_idx[0]
    dt_sec = (event_samples[next_idx] - event_samples[start_idx]) / sfreq
    intervals["start_sample"].append(event_samples[start_idx])
    intervals["target_sample"].append(event_samples[next_idx])
    intervals["target_id"].append(event_ids[next_idx])
    intervals["dt_sec"].append(dt_sec)

print(f"Found {len(intervals['dt_sec'])} cue to target intervals.")
print(f"The average timing cue to target is {np.mean(intervals['dt_sec'])} seconds.")

## 2. image to mask 
is_80 = event_ids == 80
start_indices = np.where(is_target)[0]

intervals = {"start_sample":[],
             "start_id":[],
            "target_sample":[],
            "target_id": [],
            "dt_sec": []}

for start_idx in start_indices:
    next_80_idx = np.where(is_80 & (np.arange(len(events)) > start_idx))[0]
    if next_80_idx.size == 0:
        continue
    next_idx = next_80_idx[0]
    dt_sec = (event_samples[next_idx] - event_samples[start_idx]) / sfreq
    intervals["start_sample"].append(event_samples[start_idx])
    intervals["start_id"].append(event_ids[start_idx]),
    intervals["target_sample"].append(event_samples[next_idx])
    intervals["target_id"].append(event_ids[next_idx])
    intervals["dt_sec"].append(dt_sec)

print(f"Found {len(intervals['dt_sec'])} target to mask intervals.")

short_ids = [11, 12, 13, 14]
long_ids = [21, 22, 23, 24]

short_dt = [sec for sec_idx, sec in enumerate(intervals['dt_sec']) if intervals['start_id'][sec_idx] in short_ids]
long_dt = [sec for sec_idx, sec in enumerate(intervals['dt_sec']) if intervals['start_id'][sec_idx] in long_ids]
print(f"The average timing from target to short mask is {np.mean(short_dt)} seconds.")
print(f"The average timing from target to long mask is {np.mean(long_dt)} seconds.")


##### Check event numbers ######
event_dict = {ck: events_ids[ck] for ck in condition_keys}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.5,
    preload=True,
    picks = 'eeg'
)

s = pd.Series(epochs.events[::, 2])
print(f"Total trial values:   {len(s)}")
print(f"Unique ids:  {s.nunique()}")
print(f"\nID counts:\n{s.value_counts()}")
