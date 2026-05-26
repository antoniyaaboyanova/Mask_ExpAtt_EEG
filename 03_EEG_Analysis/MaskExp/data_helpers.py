import numpy as np 
import os

# Events functions 
def get_main_events():
    """Return the main event identifiers for the experiment.

    Returns
    -------
    events : ndarray
        Array of event identifiers which cross mask and expectation conditions.

    Identifiers explained
    -------
    10s -> early mask, 20s -> late mask
    in the tens -> neutral condition
    in the hundrends -> expected condition 
    in the two hunderends -> unexpected condition
    """
    base = np.array([1,2,3,4])
    mask_conds = np.append(base + 10, base + 20)
    events = np.append(mask_conds, mask_conds + 100)
    events = np.append(events, mask_conds + 200)
    return events

def extract_expectations(eeg, ids):
    """Extract neutral, expected, and unexpected condition epochs from EEG data.

    Parameters
    ----------
    eeg : ndarray
        EEG epochs, shape (n_trials, n_sensors, n_time).
    ids : ndarray
        Condition identifiers for each trial, shape (n_trials,).

    Returns
    -------
    neutral : ndarray
        Sorted EEG epochs for the neutral condition.
    expected : ndarray
        Sorted EEG epochs for the expected condition.
    unexpected : ndarray
        Sorted EEG epochs for the unexpected condition.
    """
    neutral = np.array([11,12,13,14,21,22,23,24])
    expected = neutral + 100
    unexpected = neutral + 200

    conditions = [neutral, expected, unexpected]
    sorted_conditions = []

    for condition in conditions:
        condition_mask = np.isin(ids, condition)
        condition_ids = ids[condition_mask]
        condition_eeg = eeg[condition_mask]

        if condition_ids.size == 0:
            raise ValueError(f"No trials found for condition {condition[:5]}...")

        min_trials = np.min(np.unique(condition_ids, return_counts=True)[1])
        sorted_conditions.append(sort_eeg(condition_eeg, condition_ids, min_trials=min_trials))

    neutral, expected, unexpected = sorted_conditions
    return neutral, expected, unexpected

def condition_events(allowed_events, rename_cond="stims"):
    """Create a mapping of event labels based on allowed events and naming style.

    Parameters
    ----------
    allowed_events : iterable
        Event numbers to include in the mapping.
    rename_cond : str, optional
        Naming style for the returned labels. Must be one of "mask", "mask/cue", or "stims".

    Returns
    -------
    rename_dict : dict
        Mapping from original event labels to renamed labels.
    """
    valid_conds = {"mask", "mask/cue", "stims"}
    
    if rename_cond not in valid_conds:
        raise ValueError(f"rename_cond must be one of {valid_conds}, got '{rename_cond}'")

    rename_dict = {}

    for event_num in allowed_events:
        s = str(event_num)
        key = f"Stimulus/S {event_num}" if len(s) <= 2 else f"Stimulus/S{event_num}"

        if rename_cond == "stims":
            rename_dict[key] = key
            continue


        elif rename_cond == "mask":
            second_digit = int(s[0]) if len(s) == 2 else int(s[1])
            mask = "Short" if second_digit == 1 else "Long"
            rename_dict[key] = f"{mask}"
        
        elif rename_cond == "mask/cue":
            second_digit = int(s[0]) if len(s) == 2 else int(s[1])
            
            mask = "Short" if second_digit == 1 else "Long"

            if len(s) != 2:
                first_digit = int(s[0])
                if first_digit == 1:
                    cue = "Expected"
                elif first_digit == 2:
                    cue = "Unexpected"
                else:
                    cue = None
            else:
                cue = "Neutral"

            rename_dict[key] = f"{mask}/{cue}"

    return rename_dict

    
def sort_eeg(eeg, ids, min_trials):
    """Sort EEG epochs by unique condition identifiers and limit trials per condition.

    Parameters
    ----------
    eeg : ndarray
        EEG epochs, shape (n_trials, n_channels, n_time).
    ids : ndarray
        Condition identifiers for each trial.
    min_trials : int
        Number of trials to retain for each unique condition.

    Returns
    -------
    sorted_eeg : ndarray
        EEG array shaped (n_conditions, min_trials, n_channels, n_time).
    """
    tot_trials, channels, time = eeg.shape
    unique_ids = np.unique(ids)
    image_conditions = len(unique_ids)

    sorted_eeg = np.full((image_conditions, min_trials, channels, time), np.nan)
    for uid_idx, uid in enumerate(unique_ids):
        uid_mask = np.isin(ids, uid)
        eeg_uid = eeg[uid_mask]
        indexes = np.arange(0, len(eeg_uid))
        np.random.shuffle(indexes)
        
        sorted_eeg[uid_idx] = eeg_uid[indexes[0:min_trials]]

    return sorted_eeg

def get_sorted_cues(ids, eeg, unique_, counts_, start, end):
    block_ids = unique_[start:end]
    min_trials = counts_[start:end].min()
    
    mask = np.isin(ids, block_ids)
    selected_ids = ids[mask]
    
    sorted_block = sort_eeg(eeg, selected_ids, min_trials=min_trials)
    return sorted_block, min_trials