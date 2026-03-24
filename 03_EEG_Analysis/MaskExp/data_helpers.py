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
    valid_conds = {"side", "side/mask", "stims"}
    
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

    return rename_dict

def condition_events_decoding(events, rename_cond="side")
