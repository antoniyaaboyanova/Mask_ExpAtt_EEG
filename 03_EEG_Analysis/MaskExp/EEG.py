
### General
import numpy as np
import pandas as pd
import os
from tqdm import tqdm 

### Preprocessing 
import mne

### Decoding
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

### Local
from ExpAtt.data_helpers import dump_data, load_data
from MaskExp.data_helpers import *
from ExpAtt.decoding_pstrials import pseudotrials_general
from ExpAtt.decoding_mvnn import apply_sigma, estimate_sigma, mvnn



PATH = "/projects/archiv/DataStore_Boyanova/Mask_ExpAtt_EEG/"
CLASSIFIER = LinearDiscriminantAnalysis()
#CLASSIFIER = LinearSVC(dual=True, penalty='l2', loss='hinge', C=1.0, max_iter=10000)

#### PREPROCESSING ####
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

def preprocess_eeg_cue(sub, tmin = -.2, tmax = 2.5, baseline = (-.2, 0), highpass = 0.1, lowpass = 100, resample = 250):
    

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

    # Epoching phase
    events, events_ids = mne.events_from_annotations(raw)
    condition_keys = list(set(rename_dict.values()))
    event_dict = {ck: events_ids[ck] for ck in condition_keys}

    # Epoch around trigger 1 (cue onset)
    cue_event_id = {'cue': events_ids['Stimulus/S  1']}
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

    # Get the target codes 
    target_codes = np.sort([events_ids[name] for name in condition_keys if name in events_ids and name != '1'])
    sfreq = raw.info['sfreq']

    # Time axis for epochs
    times = epochs.times  
    cropped_epochs = []
    cropped_ids = []

    # Now, for each epoch, find the target trigger and crop to -200ms to 1000ms relative to target
    for i, epoch in enumerate(epochs):
        epoch_events = events
        
        # Find events within this epoch's time window
        epoch_start_sample = epochs.events[i, 0] + int(tmin * sfreq)
        epoch_end_sample = epochs.events[i, 0] + int(tmax * sfreq)
        epoch_event_mask = (epoch_events[:, 0] >= epoch_start_sample) & (epoch_events[:, 0] <= epoch_end_sample)
        epoch_events_in = epoch_events[epoch_event_mask] # there should be 3 (cue, target_onset, mask_onset)
        
        # Find the target event after the cue
        cue_sample = epochs.events[i, 0]
        target_events = epoch_events_in[(epoch_events_in[:, 0] > cue_sample) & np.isin(epoch_events_in[:, 2], target_codes)]
        target_sample = target_events[0, 0] 
        target_id = target_events[0, 2]
        
        # Crop epoch to -200ms to 1000ms relative to target
        target_time_in_epoch = (target_sample - cue_sample) / sfreq
        crop_tmin = target_time_in_epoch - 0.2
        crop_tmax = target_time_in_epoch + 1.0

        
        duration = crop_tmax - crop_tmin
        n_samples = int(round(duration * sfreq))  # 300 for 1.2 s at 250 Hz

        start_idx = np.searchsorted(times, crop_tmin, side='left')
        end_idx = start_idx + n_samples

        if end_idx > epoch.shape[1]:
            continue  # or handle edge trials
        cropped_epoch = epoch[:, start_idx:end_idx]

        # Apply cropping
        cropped_epoch = epoch[:, start_idx:end_idx]
        cropped_epochs.append(cropped_epoch)
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


#### DECODING ####
def decode_general(sub, epoched_ver="cue", imgPerm=10, testsize=0.2, group_size=4, whitening=False):
    """
    Run temporal generalization decoding on EEG pseudotrials.

    Parameters
    ----------
    sub : int
        Subject number.

    PATH : str
        Base project path.

    epoched_ver : str
        Epoching version ("cue" or other).

    imgPerm : int
        Number of image permutations.

    testsize : float
        Fraction of pseudotrials used for testing.

    group_size : int
        Number of conditions processed together.

    whitening : bool
        Whether to apply MVNN whitening.

    CLASSIFIER : sklearn classifier instance
        Classifier object with fit/predict methods.

    Returns
    -------
    decAcc : ndarray
    decAcc_e : ndarray
    decAcc_u : ndarray
    """

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    all_events = get_main_events()

    if epoched_ver == "cue":
        subject_path = os.path.join(
            PATH,
            "eeg_epoched",
            f"eeg_MaskExp_{sub:04d}_cue_target.pickle"
        )
    else:
        subject_path = os.path.join(
            PATH,
            "eeg_epoched",
            f"eeg_MaskExp_{sub:04d}.pickle"
        )

    subject_data = load_data(subject_path)

    ids_ = subject_data["ids"]
    eeg_ = subject_data["eeg"]

    n_conditions = len(np.unique(ids_)) // 3
    _, n_sensors, n_time = eeg_.shape

    # ------------------------------------------------------------------
    # Storage arrays
    # ------------------------------------------------------------------
    decAcc = np.full((n_conditions, n_conditions, n_time, n_time), np.nan)

    decAcc_e = np.full_like(decAcc, np.nan)
    decAcc_u = np.full_like(decAcc, np.nan)

    indexes = np.arange(n_conditions)

    # ------------------------------------------------------------------
    # Separate trials
    # ------------------------------------------------------------------
    n, e, u = extract_expectations(eeg_, ids_)

    # ------------------------------------------------------------------
    # Image permutation loop
    # ------------------------------------------------------------------
    for _ in tqdm(range(imgPerm), desc="Image permutations"):

        # Neutral pseudotrials
        pstrials, pk = pseudotrials_general(n)

        # Match expected to unexpected trial counts
        u_trials = u.shape[1]
        e_trials = e.shape[1]

        e_match = e[:, np.random.choice(
            np.arange(e_trials),
            size=u_trials,
            replace=False
        )]

        # Expected/unexpected pseudotrials
        pstrials_e, pk_e = pseudotrials_general(e_match)
        pstrials_u, pk_u = pseudotrials_general(u)

        # --------------------------------------------------------------
        # Cross-validation params
        # --------------------------------------------------------------
        n_conditions, n_pstrials, n_sensors, n_time = pstrials.shape
        _, n_pstrials_e, _, _ = pstrials_e.shape

        n_test = int(n_pstrials * testsize)
        n_test_e = int(n_pstrials_e * testsize)

        cvs = int(n_pstrials / n_test)

        ps_ixs = np.arange(n_pstrials)
        ps_ixs_e = np.arange(n_pstrials_e)

        # --------------------------------------------------------------
        # Cross-validation loop
        # --------------------------------------------------------------
        for cv in tqdm(range(cvs), desc="Cross-Validation", leave=False):

            test_ix = np.arange(n_test) + (cv * n_test)
            test_ix_e = np.arange(n_test_e) + (cv * n_test_e)

            train_ix = np.delete(ps_ixs.copy(), test_ix)

            ps_train = pstrials[:, train_ix, :, :]
            ps_test = pstrials[:, test_ix, :, :]

            ps_test_e = pstrials_e[:, test_ix_e, :, :]
            ps_test_u = pstrials_u[:, test_ix_e, :, :]

            # ----------------------------------------------------------
            # Condition groups
            # ----------------------------------------------------------
            for start in range(0, n_conditions, group_size):

                group = indexes[start:start + group_size]

                # ------------------------------------------------------
                # MVNN whitening
                # ------------------------------------------------------
                if whitening:

                    sigma_inv = estimate_sigma(ps_train[group])

                    ps_test_e_w = apply_sigma(
                        ps_test_e[group],
                        sigma_inv
                    )

                    ps_test_u_w = apply_sigma(
                        ps_test_u[group],
                        sigma_inv
                    )

                    ps_train_w, ps_test_w = mvnn(
                        ps_train[group],
                        ps_test[group],
                        group_size,
                        n_sensors,
                        n_time
                    )

                else:

                    ps_train_w = ps_train[group]
                    ps_test_w = ps_test[group]

                    ps_test_e_w = ps_test_e[group]
                    ps_test_u_w = ps_test_u[group]

                # ------------------------------------------------------
                # Pairwise decoding
                # ------------------------------------------------------
                for i, cA in enumerate(group):

                    for cB in group[i + 1:]:

                        for t in range(n_time):

                            # ------------------------------
                            # Training data
                            # ------------------------------
                            train_x = np.array([
                                ps_train_w[cA % group_size, :, :, t],
                                ps_train_w[cB % group_size, :, :, t]
                            ])

                            train_x = np.reshape(
                                train_x,
                                (len(train_ix) * 2, n_sensors)
                            )

                            train_y = np.array(
                                [1] * len(train_ix) +
                                [2] * len(train_ix)
                            )

                            # ------------------------------
                            # Neutral test
                            # ------------------------------
                            test_x = np.array([
                                ps_test_w[cA % group_size],
                                ps_test_w[cB % group_size]
                            ])

                            test_x = np.reshape(
                                test_x,
                                (len(test_ix) * 2, n_sensors, n_time)
                            )

                            test_y = np.array(
                                [1] * len(test_ix) +
                                [2] * len(test_ix)
                            )

                            # ------------------------------
                            # Expected test
                            # ------------------------------
                            expected_test_x = np.array([
                                ps_test_e_w[cA % group_size],
                                ps_test_e_w[cB % group_size]
                            ])

                            expected_test_x = np.reshape(
                                expected_test_x,
                                (len(test_ix_e) * 2, n_sensors, n_time)
                            )

                            expected_test_y = np.array(
                                [1] * len(test_ix_e) +
                                [2] * len(test_ix_e)
                            )

                            # ------------------------------
                            # Unexpected test
                            # ------------------------------
                            unexpected_test_x = np.array([
                                ps_test_u_w[cA % group_size],
                                ps_test_u_w[cB % group_size]
                            ])

                            unexpected_test_x = np.reshape(
                                unexpected_test_x,
                                (len(test_ix_e) * 2, n_sensors, n_time)
                            )

                            unexpected_test_y = np.array(
                                [1] * len(test_ix_e) +
                                [2] * len(test_ix_e)
                            )

                            # ------------------------------
                            # Train classifier
                            # ------------------------------
                            classifier = CLASSIFIER
                            classifier.fit(train_x, train_y)

                            # ------------------------------
                            # Temporal generalization
                            # ------------------------------
                            for tt in range(n_time):

                                # Neutral
                                pred_y = classifier.predict(
                                    test_x[:, :, tt]
                                )

                                acc_score = accuracy_score(
                                    test_y,
                                    pred_y
                                )

                                decAcc[cA, cB, t, tt] = np.nansum([
                                    decAcc[cA, cB, t, tt],
                                    acc_score
                                ])

                                # Expected
                                pred_y = classifier.predict(
                                    expected_test_x[:, :, tt]
                                )

                                acc_score = accuracy_score(
                                    expected_test_y,
                                    pred_y
                                )

                                decAcc_e[cA, cB, t, tt] = np.nansum([
                                    decAcc_e[cA, cB, t, tt],
                                    acc_score
                                ])

                                # Unexpected
                                pred_y = classifier.predict(
                                    unexpected_test_x[:, :, tt]
                                )

                                acc_score = accuracy_score(
                                    unexpected_test_y,
                                    pred_y
                                )

                                decAcc_u[cA, cB, t, tt] = np.nansum([
                                    decAcc_u[cA, cB, t, tt],
                                    acc_score
                                ])

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    decAcc /= (cvs * imgPerm)
    decAcc_e /= (cvs * imgPerm)
    decAcc_u /= (cvs * imgPerm)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_dir = os.path.join(
        PATH,
        "eeg_decoding",
        epoched_ver
    )

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"eeg_decoding_neutral_{sub:04d}.npy"), decAcc)
    np.save(os.path.join(save_dir, f"eeg_decoding_expected_{sub:04d}.npy"),decAcc_e)
    np.save(os.path.join(save_dir, f"eeg_decoding_unexpected_{sub:04d}.npy"), decAcc_u)

    return decAcc, decAcc_e, decAcc_u