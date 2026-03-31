import numpy as np 
import os

# Events functions 
def get_main_events():
    base = np.array([1,2,3,4,5,6,7,8])
    mask_conds = np.append(base + 10, base + 20)
    events = np.append(mask_conds, mask_conds + 100)
    events = np.append(events, mask_conds + 200)
    return events

def condition_events(allowed_events, rename_cond="side"):
    valid_conds = {"side", "side/mask", "stims", "side/mask/cue"}
    
    if rename_cond not in valid_conds:
        raise ValueError(f"rename_cond must be one of {valid_conds}, got '{rename_cond}'")

    rename_dict = {}

    for event_num in allowed_events:
        s = str(event_num)
        key = f"Stimulus/S {event_num}" if len(s) <= 2 else f"Stimulus/S{event_num}"

        if rename_cond == "stims":
            rename_dict[key] = key
            continue

        # shared logic
        base_digit = event_num % 10
        side = "Left" if base_digit in [1, 2, 3, 4] else "Right"

        if rename_cond == "side":
            rename_dict[key] = side

        elif rename_cond == "side/mask":
            second_digit = int(s[0]) if len(s) == 2 else int(s[1])
            mask = "Short" if second_digit == 1 else "Long"
            rename_dict[key] = f"{side}/{mask}"
        
        elif rename_cond == "side/mask/cue":
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

            rename_dict[key] = f"{side}/{mask}/{cue}"

    return rename_dict

def condition_events_decoding(rename_cond="side"):
    
    allowed_events = get_main_events()
    valid_conds = {"side", "side/mask", "side/mask/cue-category", "mask/cue-category"}
    if rename_cond not in valid_conds:
            raise ValueError(f"rename_cond must be one of {valid_conds}, got '{rename_cond}'")

    rename_dict = {}

    for event_num in allowed_events:
        s = str(event_num)
        key = event_num

        # shared logic
        side = event_num % 10

        if rename_cond == "side":
            rename_dict[key] = side

        elif rename_cond == "side/mask":
            second_digit = int(s[0]) if len(s) == 2 else int(s[1])
            mask = second_digit
            rename_dict[key] = int(f"{mask}{side}")
        
        elif rename_cond == "side/mask/cue-category":
            mask = int(s[0]) if len(s) == 2 else int(s[1])
            cue = 0 if len(s) == 2 else int(s[0])
            #side = (side + 1) // 2 ### this would be the logic if I fix the triggers
            block = (side - 1) // 4          # which block of 4 you're in
            pos = (side - 1) % 4 + 1         # position inside the block (1–4)

            if pos in (1, 4):
                side = block * 2 + 1
            else:
                side = block * 2 + 2
            rename_dict[key] = int(f"{cue}{mask}{side}")
            
                
        elif rename_cond == "mask/cue-category":
            mask = int(s[0]) if len(s) == 2 else int(s[1])
            cue = 0 if len(s) == 2 else int(s[0])
            #side = (side + 1) // 2 ### this would be the logic if I fix the triggers
            block = (side - 1) // 4          # which block of 4 you're in
            pos = (side - 1) % 4 + 1 
            if pos in (1, 4):
                side = block * 2 + 1
            else:
                side = block * 2 + 1
            rename_dict[key] = int(f"{cue}{mask}{side}")
    
    return rename_dict
    
def sort_eeg(eeg, ids, min_trials):
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