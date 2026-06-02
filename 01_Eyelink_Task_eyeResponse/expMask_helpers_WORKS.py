import numpy as np
import random
import os
import pandas as pd 
from psychopy import visual, core, event, gui
from eyelink_helpers import *
from instructions import (
    INTRO_TEXT,
    IDENTITY_TEXT,
    LOCATION_TEXT,
    BREAK_TEXT,
    START_TEXT,
    PRACTICE_TEXT
)

# =====================================================
# Trial Functions
# =====================================================
def create_cue_dynam(highProb=0.7, lowProb=0.3, neutral=1.0, trials_per_cue=40, trial_per_neutral=32):
    
    
    cue_data = {"cue_names": [".\\cues\\Sea_Animal.png",  ".\\cues\\Water_Vehicle.png",  ".\\cues\\Neutral.png"],
                "cue_highProb_cats": [["dolphin", "whale"], ["speedboat", "submarine"], 
                                    ["dolphin", "whale", "speedboat", "submarine"]],
                
                "cue_lowProb_cats": [["speedboat", "submarine"], ["dolphin", "whale"],
                                    ["dolphin", "whale", "speedboat", "submarine"]]}

    cue_data["cue_highProb"] = []
    cue_data["cue_lowProb"] = []
    
    for cue_id, cue in enumerate(cue_data["cue_names"]):
        if cue != ".\\cues\\Neutral.png":
            cue_data["cue_highProb"].append([np.round(highProb / len(cue_data["cue_highProb_cats"][cue_id]), 2)] * len(cue_data["cue_highProb_cats"][cue_id]))
            cue_data["cue_lowProb"].append([np.round(lowProb / len(cue_data["cue_lowProb_cats"][cue_id]), 2)] * len(cue_data["cue_lowProb_cats"][cue_id]))
        
        else:
            cue_data["cue_highProb"].append([np.round(neutral / len(cue_data["cue_highProb_cats"][cue_id]), 3)] * len(cue_data["cue_highProb_cats"][cue_id]))
            cue_data["cue_lowProb"].append([np.round(neutral / len(cue_data["cue_lowProb_cats"][cue_id]), 3)] * len(cue_data["cue_highProb_cats"][cue_id]))
            
    cue_data["high_prob_trials"] = [
    np.array(x) * (trial_per_neutral if cue_data["cue_names"][idx] == ".\\cues\\Neutral.png" else trials_per_cue)
    for idx, x in enumerate(cue_data["cue_highProb"])]
    
    cue_data["low_prob_trials"] =  [
    np.array(x) * (trial_per_neutral if cue_data["cue_names"][idx] == ".\\cues\\Neutral.png" else trials_per_cue)
    for idx, x in enumerate(cue_data["cue_lowProb"])]
    
    return cue_data

def assign_trigger(row, late=0.100):

    # ---- MASK ----
    if row['mask_ISI'] == 0.0165:
        mask_code = 10
    elif row['mask_ISI'] == late:
        mask_code = 20
    else:
        raise ValueError("Unknown mask type")

    # ---- SIDE ----
    if row['target_loc'] == 'L':
        side_code = 0
    elif row['target_loc'] == 'R':
        side_code = 100
    else:
        raise ValueError("Unknown side")

    # ---- EXPECTATION ----
    if row['expectation'] == 'neutral':
        exp_code = 0
    elif row['expectation'] == 'expected':
        exp_code = 30
    elif row['expectation'] == 'unexpected':
        exp_code = 60
    else:
        raise ValueError("Unknown expectation")

    # ---- IMAGE ----
    image_code = row['image_index']  # 1–8

    return mask_code + side_code + exp_code + image_code

def create_changes(list_length, prec):
    # Number of instances to set as True (2%)
    num_true = int(list_length * prec)

    # Create lists of False values
    catch = [False] * list_length

    # Randomly select indices for fixing and imaging channels
    catch_indices = random.sample(range(list_length), num_true)

    for idx in catch_indices:
        catch[idx] = True
    
    return np.array(catch)

def build_constrained_order(df, rng=None, max_unexpected_run=1, max_attempts=1000):
    
    if rng is None:
        rng = random.Random()

    for _ in range(max_attempts):
        remaining = df.copy().reset_index(drop=True)
        ordered_rows = []
        last_target = None
        unexpected_run = 0
        failed = False

        while len(remaining) > 0:

            # Build valid candidates mask
            valid_mask = np.ones(len(remaining), dtype=bool)

            # Rule 1 — no same target twice in a row
            if last_target is not None:
                valid_mask &= (remaining["target"].astype(str) != str(last_target))

            # Rule 2 — max consecutive unexpected trials
            if unexpected_run >= max_unexpected_run:
                valid_mask &= (remaining["expectation"].astype(str) != "unexpected")

            # Integer positions of valid rows within remaining
            valid_positions = np.where(valid_mask)[0]

            if len(valid_positions) == 0:
                failed = True
                break

            # Pick a random valid position
            chosen_pos = valid_positions[rng.randrange(len(valid_positions))]
            row = remaining.iloc[chosen_pos]

            ordered_rows.append(row)

            # Update state
            last_target = str(row["target"])
            unexpected_run = unexpected_run + 1 if str(row["expectation"]) == "unexpected" else 0

            # Drop by integer position and reset index
            remaining = remaining.drop(remaining.index[chosen_pos]).reset_index(drop=True)

        if not failed:
            return pd.DataFrame(ordered_rows).reset_index(drop=True)

    raise RuntimeError(
        f"Could not build a valid trial order after {max_attempts} attempts. "
        "Consider relaxing the constraints."
    )

def create_block_trials(stim_path, cue_data, random_seed, long_isi=0.1, identity_catch=1.0, location_catch=0.0):
    rng = random.Random(random_seed)
    long_isi = 0.1
    categories = os.listdir(stim_path)
    stimuli = []
    for cat in categories:
        cat_path = os.path.join(stim_path, cat)
        files = os.listdir(cat_path)
        stimuli.extend([f".\\stimuli\\{cat}\\{x}" for x in files])

    stimuli = np.array(stimuli)
    stimuli = stimuli[np.argsort(stimuli)]

    data = {
        "target_id": [],
        "distractor_id": [],
        "distractor_selection_id": [],
        "target": [],
        "distractor": [],
        "distractor_selection": [],
        "distractor_selection_loc": [],
        "target_selection_loc": [],
        "expectation": [],
        "mask_ISI": [],
        "cue": [],
        "target_name": [],
        "target_cat": [],
        "target_loc": []
    }

    mask_type = [0.0165, long_isi]
    distractors = stimuli[np.char.count(stimuli, "mask") > 0]
    distractor_ids = np.arange(len(stimuli))[np.isin(stimuli, distractors)]
    local_distractor_ids = np.arange(0, len(distractors))

    mapping = {
        0: 1,
        1: 2,
        14: 3,
        15: 4,
        6: 5,
        7: 6,
        10: 7,
        11: 8
    }

    def add_trials(targets, target_ids, n_trials, mask, cue, expectation_label):
        """
        Add n_trials rows for a given set of targets, guaranteeing all
        data-dict lists grow by exactly n_trials entries.

        The target images are cycled so each gets as equal exposure as
        possible, independent of whether n_trials is divisible by the
        number of images or by 2.
        """
        n_targets = len(targets)
        if n_targets == 0 or n_trials == 0:
            return

        target_names = [x.split("\\")[-1] for x in targets]
        target_categories = [x.split("_")[0] for x in target_names]

        # Build per-trial target assignments by cycling through images
        # so every image appears as evenly as possible.
        repeats = [n_trials // n_targets + (1 if i < n_trials % n_targets else 0)
                   for i in range(n_targets)]

        trial_target_ids   = []
        trial_targets      = []
        trial_dist_sel_ids = []
        trial_dist_sels    = []
        trial_names        = []
        trial_cats         = []

        for i, r in enumerate(repeats):
            trial_target_ids.extend([target_ids[i]] * r)
            trial_targets.extend([targets[i]] * r)
            # paired distractor-selection is the "other" image (flipped)
            paired_idx = (i + 1) % n_targets
            trial_dist_sel_ids.extend([target_ids[paired_idx]] * r)
            trial_dist_sels.extend([targets[paired_idx]] * r)
            trial_names.extend([target_names[i]] * r)
            trial_cats.extend([target_categories[i]] * r)

        # Interleave L/R target locations as evenly as possible
        locs = (["L", "R"] * (n_trials // 2 + 1))[:n_trials]

        random_distractors = np.random.choice(local_distractor_ids, n_trials)

        data["target_id"].extend(trial_target_ids)
        data["target"].extend(trial_targets)
        data["distractor_selection_id"].extend(trial_dist_sel_ids)
        data["distractor_selection"].extend(trial_dist_sels)
        data["target_loc"].extend(locs)
        data["target_name"].extend(trial_names)
        data["target_cat"].extend(trial_cats)
        data["distractor_id"].extend([distractor_ids[x] for x in random_distractors])
        data["distractor"].extend([distractors[x] for x in random_distractors])
        data["expectation"].extend([expectation_label] * n_trials)
        data["mask_ISI"].extend([mask] * n_trials)
        data["cue"].extend([cue] * n_trials)

    for cue_id, cue in enumerate(cue_data["cue_names"]):
        high_cats = np.array(cue_data["cue_highProb_cats"][cue_id])
        low_cats  = np.array(cue_data["cue_lowProb_cats"][cue_id])
        is_neutral = (cue == '.\\cues\\Neutral.png')

        # ---- LOW-PROBABILITY / UNEXPECTED trials (non-neutral cues only) ----
        if not is_neutral:
            for i, l_cat in enumerate(low_cats):
                l_cat_stim = stimuli[np.char.count(stimuli, l_cat) > 0]
                targets    = l_cat_stim[np.char.count(l_cat_stim, "mask") == 0]
                target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]

                for mask in mask_type:
                    # Round to nearest even number so L/R split is always exact
                    n_trials = int(cue_data["low_prob_trials"][cue_id][i])
                    if n_trials % 2 != 0:
                        n_trials += 1
                    add_trials(targets, target_ids, n_trials, mask, cue, "unexpected")

        # ---- HIGH-PROBABILITY / EXPECTED (or NEUTRAL) trials ----
        for i, h_cat in enumerate(high_cats):
            h_cat_stim = stimuli[np.char.count(stimuli, h_cat) > 0]
            targets    = h_cat_stim[np.char.count(h_cat_stim, "mask") == 0]
            target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]

            expectation_label = "neutral" if is_neutral else "expected"

            for mask in mask_type:
                n_trials = int(cue_data["high_prob_trials"][cue_id][i])
                if n_trials % 2 != 0:
                    n_trials += 1
                add_trials(targets, target_ids, n_trials, mask, cue, expectation_label)

    # ---- Derived columns (computed after all rows are collected) ----
    n_total = len(data["target"])

    data["target_selection_loc"]    = [np.random.choice(["up", "down"], p=[0.5, 0.5]) for _ in range(n_total)]
    data["distractor_selection_loc"] = ["down" if x == "up" else "up" for x in data["target_selection_loc"]]
    data["distractor_loc"]           = ["R" if x == "L" else "L" for x in data["target_loc"]]

    for key in data.keys():
        print(f"{key} : {len(data[key])}")

    df = pd.DataFrame(data)
    df['image_index'] = df['target_id'].map(mapping)
    df['trigger']     = df.apply(assign_trigger, axis=1)
    df["identity_catch"] = create_changes(len(df), identity_catch)
    df["location_catch"] = create_changes(len(df), location_catch)

    df = build_constrained_order(df, rng=rng)

    return df, stimuli

# =====================================================
# PsychoPy functions
# =====================================================
def make_text_stim(win, text):
    return visual.TextStim(
        win,
        text=text,
        height=32,          
        pos=(0, 0),
        color=[-1, -1, -1],
        units="pix",
        wrapWidth=856) 

    
def draw_and_wait(win, draw_func):
    event.clearEvents(eventType="keyboard")

    draw_func()
    win.flip()

    keys = event.waitKeys(keyList=["space", "escape"])

    if "escape" in keys:
        win.close()
        core.quit()
    
def draw_and_wait(win, draw_func):
    event.clearEvents(eventType="keyboard")
    while True:
        draw_func()
        win.flip()

        keys = event.getKeys(keyList=["space", "escape"])
        if "space" in keys:
            break
        if "escape" in keys:
            win.close()
            core.quit()

def run_block(win,
    el_tracker,
    image_data,
    stimuli,

    # stimuli & layout
    cue_stim,
    fixation_cross,
    fixation_arrows,        # used only on location catch trials
    mask_pool,
    pos_left,
    pos_right,

    # timing
    cue_duration,
    precue_fix,
    postcue_fix,
    image_duration,
    id_response,            # used on identity catch trials
    loc_response,           # used on location catch trials
    preresp_fix,
    postresp_fix,
    n_masks_per_trial,

    # bookkeeping
    participant_num,
    run_num,
    output_dir,
    output_filename,
    session_folder,
    edf_fname,
    *,
    break_number=80,
    practice=False,
    dummy=False,
    instruct='ide'):

    # =====================================================
    # TEXT CREATION
    # =====================================================
    intro_text = make_text_stim(win, INTRO_TEXT)
    
    if instruct == "ide":
        task_text = make_text_stim(win, IDENTITY_TEXT)
    elif instruct == "loc":
        task_text = make_text_stim(win, LOCATION_TEXT)
        
    start_text = make_text_stim(win, START_TEXT)
    stimuli_title = visual.TextStim(
        win, text="The stimuli you will see",
        height=40, pos=(0, 400), color=[-1, -1, -1],
        units="pix", wrapWidth=856)
    stimuli_image = visual.ImageStim(
        win, image="instructions_stimuli.png",
        size=(800, 600), pos=(0, 0), units='pix')
    cues_image = visual.ImageStim(
        win, image="instructions_cues.png",
        size=(800, 400), pos=(0, 0), units='pix')
    cues_title = visual.TextStim(
        win,
        text="The cues you will see. The X symbolises neutrality, meaning it holds no predictive power of what will follow.",
        height=40, pos=(0, 250), color=[-1, -1, -1],
        units="pix", wrapWidth=856)

    trial_log = []
    block_log = []
    n_trials = len(image_data)

    ecc = 200
    pos_left_selection  = (-ecc, 0)
    pos_right_selection = ( ecc, 0)
    pos_up_selection    = (0,  ecc)
    pos_down_selection  = (0, -ecc)

    # =====================================================
    # INTRODUCTION
    # =====================================================
    draw_and_wait(win, lambda: intro_text.draw())
    draw_and_wait(win, lambda: (stimuli_title.draw(), stimuli_image.draw()))
    draw_and_wait(win, lambda: (cues_title.draw(), cues_image.draw()))
    draw_and_wait(win, lambda: task_text.draw())

    if practice:
        print(f"There will be {n_trials} trials in the practice")
        practice_text = make_text_stim(win, PRACTICE_TEXT)
        draw_and_wait(win, lambda: practice_text.draw())

    draw_and_wait(win, lambda: start_text.draw())
    
    # ── before trial loop ────────────────────────────────────────────────────────
    if el_tracker is not None:
        el_tracker.startRecording(1, 1, 1, 1)
        pylink.msecDelay(100)   # single settle wait, once per block

    global_clock = core.Clock()
    experiment_start_time = global_clock.getTime()

    for i in range(n_trials):

        # =====================================================
        # INTERIM BREAK
        # =====================================================
        if (i + 1) % break_number == 0 and i != n_trials - 1:
            print("Participant is having a break...")
            event.clearEvents(eventType="keyboard")

            # Accuracy over whichever catch type appeared in this block
            id_trials  = [t["correct_id"]  is True for t in block_log if t["identity_catch"]]
            loc_trials = [t["correct_loc"] is True for t in block_log if t["location_catch"]]
            acc_trials = id_trials or loc_trials
            acc = np.mean(acc_trials) * 100 if acc_trials else 0

            break_text = visual.TextStim(
                win,
                text=(
                    f"Please take a short break (1–2 minutes).\n\n"
                    f"Accuracy: {acc:.1f}%\n\n"
                    "Press SPACE to continue ⌨"
                ),
                height=32, color=[-1, -1, -1], wrapWidth=856)

            draw_and_wait(win, lambda: break_text.draw())
            block_log = []
            
            # ── drift check ──────────────────────────────────────────────────
            if el_tracker is not None:
                drift_check(el_tracker, win, fixation_cross, dummy=dummy)
            # ─────────────────────────────────────────────────────────────────
            
            fixation_cross.draw()
            win.flip()
            core.wait(0.5)

        # =====================================================
        # CUE PERIOD
        # =====================================================
        fixation_offset = 0.3  # seconds before postcue_fix ends that fixation disappears
        
        # ── EyeLink log for this trial ───────────────────────────
        if el_tracker is not None:
            el_tracker.sendMessage(f"TRIALID {i}")
            el_tracker.sendMessage(f"TRIGGER {image_data['trigger'][i]}")
            el_tracker.sendMessage(f"CUE {image_data['cue'][i]}")
            el_tracker.sendMessage(f"TARGET_LOC {image_data['target_loc'][i]}")
        # ─────────────────────────────────────────────────────────────────────
        
        iti_duration = np.random.uniform(precue_fix[0], precue_fix[1])
        fixation_cross.draw()
        win.flip()
        core.wait(iti_duration)

        cue_path = image_data["cue"][i]
        cue_stim.image = cue_path
        cue_stim.draw()
        win.flip()
        core.wait(cue_duration)
        
        # Fixation visible for first portion of postcue period
        fixation_cross.draw()
        win.flip()
        core.wait(postcue_fix - fixation_offset)

        # Blank screen for final portion (anticipatory period)
        win.flip()
        core.wait(fixation_offset)

        # =====================================================
        # IMAGE + MASK PERIOD
        # =====================================================
        target_id      = image_data["target_id"][i]
        distractor_id  = image_data["distractor_id"][i]
        current_target     = image_data['stim'][target_id]
        current_distractor = image_data['stim'][distractor_id]

        current_target.pos     = pos_left if image_data["target_loc"][i]     == "L" else pos_right
        current_distractor.pos = pos_left if image_data["distractor_loc"][i] == "L" else pos_right

        isi_duration = image_data["mask_ISI"][i]

        current_target.draw()
        current_distractor.draw()
        image_onset = global_clock.getTime()
        win.flip()
        if el_tracker is not None:
            el_tracker.sendMessage(f"IMG_ONSET trigger_{image_data['trigger'][i]}")
       
            
        while global_clock.getTime() < image_onset + image_duration:
            pass
        
        if el_tracker is not None:
            el_tracker.sendMessage(f"IMG_OFFSET trigger_{image_data['trigger'][i]}")

        image_offset = global_clock.getTime()
        print(f"Trial {i}: Image duration = {image_offset - image_onset}, distractor{current_distractor.image}")

        isi_onset = global_clock.getTime()
        win.flip()
        while global_clock.getTime() < isi_onset + isi_duration:
            pass

        isi_offset = global_clock.getTime()
        actual_isi_duration = isi_offset - isi_onset

        trial_masks_left  = np.random.choice(mask_pool, n_masks_per_trial, replace=False)
        trial_masks_right = np.random.choice(mask_pool, n_masks_per_trial, replace=False)

        mask_i = 0
        mask_durations = []
        if el_tracker is not None:
            el_tracker.sendMessage(f"MASK_ONSET trigger_{image_data['trigger'][i]}")
        for lm_stim, rm_stim in zip(trial_masks_left, trial_masks_right):
            lm_stim.pos = pos_left
            rm_stim.pos = pos_right
            mask_onset = global_clock.getTime()
            lm_stim.draw()
            rm_stim.draw()
            win.flip()
            while global_clock.getTime() < mask_onset + image_duration:
                pass
            mask_offset = global_clock.getTime()
            mask_durations.append(mask_offset - mask_onset)
            if mask_i == 0:
                first_mask_onset  = mask_onset
                first_mask_offset = mask_offset
            mask_i += 1

        if el_tracker is not None:
            el_tracker.sendMessage(f"MASK_OFFSET trigger_{image_data['trigger'][i]}")
        actual_mask_duration = np.mean(mask_durations)
        short_isi_st = win.flip()

        # =====================================================
        # RESPONSE PERIOD — branches on catch type
        # =====================================================
        response_id  = None;  rt_id  = None
        response_loc = None;  rt_loc = None
        correct_id   = None
        correct_loc  = None

        # ------ IDENTITY catch ------
        if image_data["identity_catch"][i]:
            
            if isi_duration == 0.1:
                preresp_fix = 0.3
            elif isi_duration == 0.0165:
                preresp_fix = 0.4
            
            win.flip()
            core.wait(preresp_fix)
            if el_tracker is not None:
                el_tracker.sendMessage(f"RESPONSE_ONSET")

            distractor_selection_id = image_data["distractor_selection_id"][i]
            target     = image_data['only_targets'][target_id]
            distractor = image_data['only_targets'][distractor_selection_id]

            target.pos     = pos_up_selection  if image_data["target_selection_loc"][i]     == "up" else pos_down_selection
            distractor.pos = pos_down_selection  if image_data["target_selection_loc"][i]     == "up" else pos_up_selection

            key_to_stim = {
                "up":  target     if image_data["target_selection_loc"][i] == "up" else distractor,
                "down": target     if image_data["target_selection_loc"][i] == "down" else distractor,
            }

            event.clearEvents(eventType='keyboard')
            target.draw()
            distractor.draw()
            t_resp_onset  = win.flip()
            response_clock = core.Clock()

            while response_id is None and response_clock.getTime() < id_response:
                keys = event.getKeys(keyList=["up", "down", "escape"], timeStamped=response_clock)
                for key, t in keys:
                    if key == "escape":
                        win.close(); core.quit()
                    if key in key_to_stim:
                        rt_id       = t
                        response_id = key_to_stim[key].image
                        break
                if response_id is not None:
                    win.flip(); break
                target.draw()
                distractor.draw()
                win.flip()

            correct_id = response_id == image_data["target"][i] if response_id else False
            win.flip()
            core.wait(postresp_fix)

        # ------ LOCATION catch ------
        elif image_data["location_catch"][i]:
            
            if isi_duration == 0.1:
                preresp_fix = 0.3
            elif isi_duration == 0.0165:
                preresp_fix = 0.4
            
            win.flip()
            core.wait(preresp_fix)
            
            if el_tracker is not None:
                el_tracker.sendMessage(f"RESPONSE_ONSET")

            key_to_loc = {"left": "L", "right": "R"}

            event.clearEvents(eventType='keyboard')
            fixation_arrows.draw()
            t_resp_onset  = win.flip()
            response_clock = core.Clock()

            while response_loc is None and response_clock.getTime() < loc_response:
                keys = event.getKeys(keyList=["left", "right", "escape"], timeStamped=response_clock)
                for key, t in keys:
                    if key == "escape":
                        win.close(); core.quit()
                    if key in key_to_loc:
                        rt_loc       = t
                        response_loc = key_to_loc[key]
                        break
                if response_loc is not None:
                    win.flip(); break
                fixation_arrows.draw()
                win.flip()

            correct_loc = response_loc == image_data["target_loc"][i] if response_loc else False

            fixation_cross.draw()
            win.flip()
            core.wait(postresp_fix)

        # =====================================================
        # PRACTICE FEEDBACK
        # =====================================================
        if practice:
            if image_data["identity_catch"][i]:
                fb = visual.TextStim(win, pos=(0, 0), height=40,
                    text=f"Response: {'Correct' if correct_id else 'Incorrect'}",
                    color="green" if correct_id else "red")
                fb.draw(); win.flip(); core.wait(2.0)

            elif image_data["location_catch"][i]:
                fb = visual.TextStim(win, pos=(0, 0), height=40,
                    text=f"Response: {'Correct' if correct_loc else 'Incorrect'}",
                    color="green" if correct_loc else "red")
                fb.draw(); win.flip(); core.wait(2.0)

        
        # ── trial end marker ─────────────────────────────────────────────────────
        if el_tracker is not None:
            el_tracker.sendMessage(
                f"TRIAL_RESULT id={response_id} loc={response_loc} "
                f"correct_id={correct_id} correct_loc={correct_loc}")
        # =====================================================
        # LOGGING
        # =====================================================
        trial_log.append({
            "participant":          participant_num,
            "block_number":         run_num,
            "trial":                i,
            "trigger":              image_data["trigger"][i],
            "cue":                  image_data["cue"][i],
            "target_name":          image_data["target_name"][i],
            "target_cat":           image_data["target_cat"][i],
            "image_onset":          image_onset,
            "image_offset":         image_offset,
            "first_mask_onset":     first_mask_onset,
            "first_mask_offset":    first_mask_offset,
            "last_mask_onset":      mask_onset,
            "last_mask_offset":     mask_offset,
            "isi_mask":             isi_duration,
            "isi_mask_actual":      actual_isi_duration,
            "expectation":          image_data["expectation"][i],
            "identity_catch":       image_data["identity_catch"][i],
            "location_catch":       image_data["location_catch"][i],
            "target_loc":           image_data["target_loc"][i],
            "target_selection_loc": image_data.get("target_selection_loc", {}).get(i, None),
            "response_id":          str(response_id),
            "rt_id":                rt_id  if rt_id  else np.nan,
            "correct_id":           correct_id  if correct_id  is not None else 'None',
            "response_loc":         str(response_loc),
            "rt_loc":               rt_loc if rt_loc else np.nan,
            "correct_loc":          correct_loc if correct_loc is not None else 'None',
        })

        block_log.append(trial_log[-1])

        print(f"Trial {i} | identity: {response_id} ({correct_id}) | location: {response_loc} ({correct_loc})")

        if not practice:
            save_data(trial_log, output_dir, output_filename)

        # Running accuracy (whichever catch type has data)
        id_trials  = [t["correct_id"]  is True for t in trial_log if t["identity_catch"]]
        loc_trials = [t["correct_loc"] is True for t in trial_log if t["location_catch"]]
        acc_trials = id_trials or loc_trials
        acc = np.mean(acc_trials) * 100 if acc_trials else 0

    # =====================================================
    # END OF BLOCK SUMMARY
    # =====================================================
    if el_tracker is not None:
        el_tracker.stopRecording()
        el_tracker.closeDataFile()
        win.flip()  # blank screen during transfer
        local_edf = os.path.join(session_folder, edf_fname + ".EDF")
        el_tracker.receiveDataFile(edf_fname + ".EDF", local_edf)
        print(f"EDF saved to: {local_edf}")

    label = "Practice complete!" if practice else f"Block {run_num + 1} complete!"
    summary_text = visual.TextStim(
        win,
        text=f"{label}\n\nAccuracy: {acc:.1f}%\n\nDo NOT press any button!\n\n The experimenter will be with you soon.",
        height=40, color=[-1, -1, -1], wrapWidth=856)
    draw_and_wait(win, lambda: summary_text.draw())
    return trial_log


def save_data(data, output_dir, filename):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir,filename), index=False)
    
def pixels_to_degrees(pixels, distance, screen_width, resolution_width):
    """
    Convert pixels to degrees of visual angle.

    Parameters:
        pixels (int): Number of pixels to convert.
        distance (float): Distance from the observer to the screen (same units as screen_width).
        screen_width (float): Physical width of the screen (same units as distance).
        resolution_width (int): Horizontal resolution of the screen (in pixels).

    Returns:
        float: Visual angle in degrees.
    """
    # Physical size of a single pixel
    pixel_size = screen_width / resolution_width

    # Physical size of the object (in the same units as screen_width)
    object_size = pixel_size * pixels

    # Calculate visual angle using the formula
    visual_angle = 2 * math.degrees(math.atan((object_size / 2) / distance))

    return visual_angle