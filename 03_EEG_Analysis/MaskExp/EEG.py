
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

PATH = "/projects/archiv/DataStore_Boyanova/Mask_ExpAtt_EEG/"
def preprocess_eeg(sub, tmin = -.2, tmax = 1.0, baseline = (-.2, 0), highpass = 0.1, lowpass = 100, resample = 250):
    
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

    return dat


def preprocess_eeg_cue(sub, tmin = -.2, tmax = 2.0, baseline = (-.2, 0), highpass = 0.1, lowpass = 100, resample = 250):
    
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

    # Epoch around trigger 1 (cue)
    cue_event_id = {'cue': events_ids['1']}
    print(f"Epoching around cue (trigger 1): between {tmin}s and {tmax}s with a baseline = {baseline}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=cue_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        picks = 'eeg'
    )

    montage = mne.channels.make_standard_montage('easycap-M1')
    epochs.set_montage(montage)

    # Now, for each epoch, find the target trigger and crop to -200ms to 1000ms relative to target
    target_codes = [events_ids[name] for name in condition_keys if name in events_ids and name != '1']
    sfreq = raw.info['sfreq']
    cropped_epochs = []
    cropped_ids = []

    for i, epoch in enumerate(epochs):
        epoch_events = mne.find_events(raw, stim_channel=None, consecutive=False, min_duration=0, 
                                       shortest_event=1, verbose=False)
        # Find events within this epoch's time window
        epoch_start_sample = epochs.events[i, 0] + int(tmin * sfreq)
        epoch_end_sample = epochs.events[i, 0] + int(tmax * sfreq)
        epoch_event_mask = (epoch_events[:, 0] >= epoch_start_sample) & (epoch_events[:, 0] <= epoch_end_sample)
        epoch_events_in = epoch_events[epoch_event_mask]
        
        # Find the first target event after the cue
        cue_sample = epochs.events[i, 0]
        target_events = epoch_events_in[(epoch_events_in[:, 0] > cue_sample) & np.isin(epoch_events_in[:, 2], target_codes)]
        if len(target_events) == 0:
            continue  # Skip if no target
        target_sample = target_events[0, 0]
        target_id = target_events[0, 2]
        
        # Crop epoch to -200ms to 1000ms relative to target
        target_time_in_epoch = (target_sample - cue_sample) / sfreq
        crop_tmin = target_time_in_epoch - 0.2
        crop_tmax = target_time_in_epoch + 1.0
        if crop_tmin < tmin or crop_tmax > tmax:
            continue  # Skip if crop is outside epoch
        cropped_epoch = epoch.crop(tmin=crop_tmin, tmax=crop_tmax)
        cropped_epochs.append(cropped_epoch.get_data()[0])  # Get the data
        cropped_ids.append(target_id)

    # Stack the cropped epochs
    if len(cropped_epochs) == 0:
        raise ValueError("No valid epochs found")
    eeg_data = np.stack(cropped_epochs)
    ids = np.array(cropped_ids)

    # Get data
    dat = { "eeg": eeg_data,
            "time": epochs.times[:int((crop_tmax - crop_tmin) * sfreq) + 1],  # Adjust time
            "ids": ids,
            "channels": epochs.ch_names}

    epoch_dir = os.path.join(PATH, "eeg_epoched")
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    dat_name = os.path.join(epoch_dir, f"eeg_MaskExp_{sub:04d}_cue_target.pickle")
        
    print(f"Total accepted trials: {dat['eeg'].shape[0]}")
    dump_data(dat, dat_name)

    return dat