
### General
import numpy as np
import pandas as pd
import os

### Preprocessing 
import mne

### Decoding
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

### Local
from ExpAtt.data_helpers import dump_data, load_data
from MaskExp.data_helpers import *

def preprocess_eeg(sub, tmin = -.2, tmax = 1.0, baseline = (-.2, 0), highpass = 0.1, lowpass = 100, resample = 250):
    
    # I don't want to see all the output
    mne.set_log_level('error')
    
    # load raw files
    folder_name = f"sub-{sub:04d}"
    sub_folder = os.path.join(PATH, "eeg", folder_name)
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
    
    print(f"Filtering: hp = {highpass}, lp = {lowpass}")
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw = raw.filter(l_freq=highpass, h_freq=lowpass, picks=eeg_picks)
    print(f"Resampling data to: {resample} Hz")
    raw = raw.resample(resample)
    print("Re-referencing...")

    raw, _ = mne.set_eeg_reference(raw, ref_channels='average', ch_type = 'eeg')

    events, events_ids = mne.events_from_annotations(raw)
    condition_keys = list(set(rename_dict.values()))
    event_dict = {ck: events_ids[ck] for ck in condition_keys}

    print(f"Epoching: between {tmin}s and {tmax}s with a baseline = {baseline}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        picks = 'eeg'
    )

    montage = mne.channels.make_standard_montage('easycap-M1')
    epochs.set_montage(montage)

    # Get data
    dat = { "eeg": epochs.get_data(),
            "time": epochs.times,
            "ids": epochs.events[::, 2],
            "channels": epochs.ch_names}

    epoch_dir = os.path.join(PATH, "eeg_epoched")
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    if epochs.get_data().shape[-1] > 301:
        dat_name = os.path.join(epoch_dir, f"eeg_MaskExp_{sub:04d}_cue.pickle")
    else:
        dat_name = os.path.join(epoch_dir, f"eeg_MaskExp_{sub:04d}.pickle")
        
    print(f"Total accepted trials: {dat['eeg'].shape[0]}")
    dump_data(dat, dat_name)
