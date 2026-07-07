import numpy as np
import random
import os
import pandas as pd 
from collections import Counter

from psychopy import visual, core, event, gui
from eyelink_helpers import *
from instructions import (
    INTRO_TEXT,
    IDENTITY_TEXT,
    LOCATION_TEXT,
    MIX_TEXT,
    BREAK_TEXT,
    START_TEXT,
    PRACTICE_TEXT
)

#=====================================================
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
    # Number of instances to set as True
    num_true = int(list_length * prec)

    # Create lists of False values
    catch = [False] * list_length

    # Randomly select indices for fixing and imaging channels
    catch_indices = random.sample(range(list_length), num_true)

    for idx in catch_indices:
        catch[idx] = True
    
    return np.array(catch)

def allocate_catch_trials(N, p=0.3, k=3, alpha=0.66):
    C = int(N * p)
    
    # ensure divisible by (k-1)*2 if needed
    while C % (k-1) != 0:
        C -= 1
    
    main = int(round(alpha * C))
    
    # enforce even split
    remainder = C - main
    other = remainder // (k - 1)
    
    # adjust if rounding broke divisibility
    main = C - other * (k - 1)
    other = [other] * (k - 1)
    
    return main, other[0], other[1]

def assign_catch_trials(df, rng=None, p=0.25, k=3, alpha=0.5, cond="identity_catch"):
    """
    Takes your existing 224-trial dataframe and:
      1. Flags 56 rows as catch trials (24 neutral, 14 expected, 14 unexpected),
         balanced across mask_ISI within each expectation condition.
      2. Reorders the sequence so catches are spaced min=2, max=7-8, mean~4 apart.
    
    Adds an cond boolean column.
    Returns a reordered dataframe (reset index).
    """
    if rng is None:
        rng = random.Random()

    df = df.copy()
    df[cond] = False

    # --- 1. Flag catch trials ---
    neut, exp, unexp = allocate_catch_trials(len(df), p, k, alpha)
    catch_counts = {'neutral': neut, 'expected': exp, 'unexpected': unexp}

    for condition, n_catches in catch_counts.items():
        cond_idx = df[df['expectation'] == condition].index.tolist()
        assert len(cond_idx) >= n_catches, \
            f"Not enough {condition} trials: need {n_catches}, have {len(cond_idx)}"

        # Balance across mask_ISI: half from 0.0165, half from 0.100
        half = n_catches // 2  # both 52 and 14 are even, so no remainder

        for isi_val, n in [(0.0165, half), (0.100, half)]:
            isi_idx = df.loc[cond_idx][df.loc[cond_idx, 'mask_ISI'] == isi_val].index.tolist()
            assert len(isi_idx) >= n, \
                f"Not enough {condition}/ISI={isi_val} trials: need {n}, have {len(isi_idx)}"
            chosen = rng.sample(isi_idx, n)
            df.loc[chosen, cond] = True

    # --- 2. Separate catches and non-catches ---
    catch_df    = df[df[cond]].sample(frac=1, random_state=rng.randint(0, 99999)).reset_index(drop=True)
    noncatch_df = df[~df[cond]].sample(frac=1, random_state=rng.randint(0, 99999)).reset_index(drop=True)

    n_catches  = len(catch_df)    # 80
    n_noncatch = len(noncatch_df) # 240

    # --- 3. Sample inter-catch gaps ---
    # gap = number of non-catch trials BEFORE each catch
    # gap in [1, 7] → total catch-to-catch distance of [2, 8], mean ~4
    gaps = _sample_gaps(
        n_catches  = n_catches,
        n_noncatch = n_noncatch,
        gap_min    = 1,
        gap_max    = 7,
        gap_mean   = 3.0,  # 3 non-catches between → distance of 4
        rng        = rng,
    )

    # --- 4. Interleave into final sequence ---
    sequence_rows = []
    nc_pointer = 0

    for i, gap in enumerate(gaps):
        # Insert `gap` non-catch trials
        for _ in range(gap):
            sequence_rows.append(noncatch_df.iloc[nc_pointer])
            nc_pointer += 1
        # Insert catch trial
        sequence_rows.append(catch_df.iloc[i])

    # Append any leftover non-catch trials at the end
    while nc_pointer < n_noncatch:
        sequence_rows.append(noncatch_df.iloc[nc_pointer])
        nc_pointer += 1

    result = pd.DataFrame(sequence_rows).reset_index(drop=True)
    return result

def _sample_gaps(n_catches, n_noncatch, gap_min, gap_max, gap_mean, rng):
    """
    Returns a list of n_catches integers in [gap_min, gap_max]
    that sum exactly to n_noncatch, distributed around gap_mean.
    """
    # Sanity check: is the target sum achievable?
    assert gap_min * n_catches <= n_noncatch <= gap_max * n_catches, (
        f"Cannot distribute {n_noncatch} non-catch trials across {n_catches} gaps "
        f"with min={gap_min}, max={gap_max}. "
        f"Feasible range: [{gap_min * n_catches}, {gap_max * n_catches}]"
    )

    for _ in range(50_000):
        gaps = [
            max(gap_min, min(gap_max, round(rng.triangular(gap_min, gap_max, gap_mean))))
            for _ in range(n_catches)
        ]
        diff = sum(gaps) - n_noncatch

        # Nudge gaps up or down to hit the exact sum
        for _ in range(5_000):
            if diff == 0:
                break
            idx = rng.randrange(n_catches)
            if diff > 0 and gaps[idx] > gap_min:
                gaps[idx] -= 1
                diff -= 1
            elif diff < 0 and gaps[idx] < gap_max:
                gaps[idx] += 1
                diff += 1

        if diff == 0:
            return gaps

    raise RuntimeError("Failed to converge on valid gap distribution.")

def validate_sequence(df, catch_cond="identity_catch"):
    catches = df[df[catch_cond]]

    print(f"Total:     {len(df)}")
    print(f"Catch {catch_cond}:     {len(catches)}  ({len(catches)/len(df)*100:.1f}%)")
    print(f"Non-catch: {len(df) - len(catches)}")

    print(f"\nCatch breakdown by expectation:")
    for cond, grp in catches.groupby('expectation'):
        c017 = (grp['mask_ISI'] == 0.0165).sum()
        c100 = (grp['mask_ISI'] == 0.100).sum()
        print(f"  {cond:11s}: total={len(grp)}  ISI=0.0165: {c017}  ISI=0.100: {c100}")

    catch_pos = df.index[df[catch_cond]].tolist()
    distances = [catch_pos[i+1] - catch_pos[i] for i in range(len(catch_pos) - 1)]

    print(f"\nCatch-to-catch distances:")
    print(f"  min={min(distances)}  max={max(distances)}  mean={np.mean(distances):.2f}")
    print(f"  distribution: {dict(sorted(Counter(distances).items()))}")

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

def get_second_selection(row):
    group1 = {0, 1, 14, 15}
    group2 = {6, 7, 10, 11}
    if row["target_id"] in group1 and row["distractor_selection_match_id"] in group1:
        pair = [(6, 7), (10, 11)][np.random.randint(2)]
    elif row["target_id"] in group2 and row["distractor_selection_match_id"] in group2:
        pair = [(0, 1), (14, 15)][np.random.randint(2)]
    else:
        pair = (np.nan, np.nan)
    return pair

def assign_locations(row):
    # randomly choose target location
    target_loc = np.random.choice(["up", "down", "left", "right"])

    match_map = {
        "up": "left",
        "left": "up",
        "down": "right",
        "right": "down",
    }

    distractor_match_loc = match_map[target_loc]

    remaining = list(
        {"up", "down", "left", "right"}
        - {target_loc, distractor_match_loc}
    )

    np.random.shuffle(remaining)

    return pd.Series({
        "target_selection_loc": target_loc,
        "distractor_match_selection_loc": distractor_match_loc,
        "distractor_2_selection_loc": remaining[0],
        "distractor_2_match_selection_loc": remaining[1],
    })

def create_block_trials(stim_path, cue_data, random_seed, long_isi=0.1, identity_catch=1.0, location_catch=0.0, k=3, alpha=0.5):
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
        "distractor_selection_match_id": [],
        "target": [],
        "distractor": [],
        "distractor_selection_match": [],
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
        
        data["distractor_selection_match_id"].extend(trial_dist_sel_ids)
        data["distractor_selection_match"].extend(trial_dist_sels)
        
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
    data["distractor_loc"] = ["R" if x == "L" else "L" for x in data["target_loc"]]
    df = pd.DataFrame(data)
    df['image_index'] = df['target_id'].map(mapping)
    df['trigger']     = df.apply(assign_trigger, axis=1)
    df = build_constrained_order(df, rng=rng)

    df[
    ["distractor_selection_2_id", "distractor_selection_2_match_id"]
    ] = df.apply(get_second_selection, axis=1, result_type="expand")

    df["distractor_selection_2"] = (
        df["distractor_selection_2_id"]
        .map(lambda x: stimuli[int(x)].split("\\")[-1] if not pd.isna(x) else np.nan)
    )

    df["distractor_selection_2_match"] = (
        df["distractor_selection_2_match_id"]
        .map(lambda x: stimuli[int(x)].split("\\")[-1] if not pd.isna(x) else np.nan)
    )


    df[[
            "target_selection_loc",
            "distractor_match_selection_loc",
            "distractor_2_selection_loc",
            "distractor_2_match_selection_loc",
        ]] = df.apply(assign_locations, axis=1)

    if identity_catch == 1.0 or identity_catch == 0.0:
        df["identity_catch"] = create_changes(len(df), identity_catch)
    else:
        df = assign_catch_trials(df, rng=rng, p=identity_catch, k=k, alpha=alpha, cond="identity_catch")
        validate_sequence(df, catch_cond="identity_catch")

    if location_catch == 1.0 or location_catch == 0.0:
        df["location_catch"] = create_changes(len(df), location_catch)
    else:
        df = assign_catch_trials(df, rng=rng, p=location_catch, k=k, alpha=alpha, cond="location_catch")
        validate_sequence(df, catch_cond="location_catch")

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
    fixation_arrows,
    resp_rect_left,
    resp_rect_right,
    mask_pool,
    pos_left,
    pos_right,
    # timing
    cue_duration,
    precue_fix,
    postcue_fix,
    image_duration,
    mask_duration,
    id_response,
    loc_response,
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
    instruct='ide',
    gaze_response=True,
    eye_used=1,
    ppd=40,
    fix_window_deg=4.0):

    # =====================================================
    # TEXT CREATION
    # =====================================================
    intro_text = make_text_stim(win, INTRO_TEXT)
    
    if instruct == "ide":
        task_text = make_text_stim(win, IDENTITY_TEXT)
    if instruct == "loc":
        task_text = make_text_stim(win, LOCATION_TEXT)
    if instruct == "mix":
        task_text = make_text_stim(win, MIX_TEXT)
        
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
    
    # Small center dots
    dot_radius = 4  # pixels

    dot_left = visual.Circle(
        win,
        radius=dot_radius,
        pos=pos_left,      # same center as left rectangle
        fillColor='black',
        lineColor='black',
        units='pix'
    )

    dot_right = visual.Circle(
        win,
        radius=dot_radius,
        pos=pos_right,     # same center as right rectangle
        fillColor='black',
        lineColor='black',
        units='pix'
    )

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
    
    # ── pre-compute EyeLink pixel coordinates (screen never changes) ─────────
    scn_w, scn_h = win.size
    cx, cy = scn_w // 2, scn_h // 2

    left_el  = (cx + pos_left[0],  cy - pos_left[1])
    right_el = (cx + pos_right[0], cy - pos_right[1])

    # ── pre-compute positions for 4afc ─────────              
    radius = 250

    loc_to_pos = {
        "right": ( radius, 0),
        "up":    (0,  radius),
        "left":  (-radius, 0),
        "down":  (0, -radius)}

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

            id_trials  = [t["correct_id"]  is True for t in block_log if t["identity_catch"]]
            loc_trials = [t["correct_loc"] is True for t in block_log if t["location_catch"]]
            acc_trials = id_trials or loc_trials
            acc_loc = np.mean(loc_trials) * 100 if loc_trials else 0
            acc_id = np.mean(id_trials) * 100 if id_trials else 0

            break_text = visual.TextStim(
                win,
                text=(
                    f"Please take a short break (1–2 minutes).\n\n"
                    f"Location: {acc_loc:.1f}%\n\n"
                    f"Identity: {acc_id:.1f}%\n\n"
                    "First, Press SPACE to continue ⌨\n\n"
                    "Then look in the centre and press ENTER"
                    
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
        
        # Select masks here to avoid potential lag
        trial_masks_left  = np.random.choice(mask_pool, n_masks_per_trial, replace=False)
        trial_masks_right = np.random.choice(mask_pool, n_masks_per_trial, replace=False)
        
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
        core.wait(postcue_fix)

        # =====================================================
        # IMAGE + MASK PERIOD
        # =====================================================
        target_id      = image_data["target_id"][i]
        distractor_id  = image_data["distractor_id"][i]
        current_target     = image_data['stim'][target_id]
        current_distractor = image_data['stim'][distractor_id]
        isi_duration = image_data["mask_ISI"][i]

        current_target.pos     = pos_left if image_data["target_loc"][i]     == "L" else pos_right
        current_distractor.pos = pos_left if image_data["distractor_loc"][i] == "L" else pos_right 
        
        if el_tracker is not None:
            el_tracker.sendMessage(f"IMG_ONSET trigger_{image_data['trigger'][i]}")
        image_onset = global_clock.getTime()  # AFTER flip
        current_target.draw()
        current_distractor.draw()
        fixation_cross.draw()
        win.flip()    

        while global_clock.getTime() < image_onset + image_duration:
            pass
        
        # win.flip()  # clear screen first
        image_offset = global_clock.getTime()  # AFTER flip
        if el_tracker is not None:
            el_tracker.sendMessage(f"IMG_OFFSET trigger_{image_data['trigger'][i]}")
            
        print(f"Trial {i}: Image duration = {image_offset - image_onset:.4f}s, distractor: {current_distractor.image}")
        isi_onset = global_clock.getTime()
        win.flip()
        while global_clock.getTime() < isi_onset + isi_duration:
            pass

        if el_tracker is not None:
            el_tracker.sendMessage(f"MASK_ONSET trigger_{image_data['trigger'][i]}")

        isi_offset = global_clock.getTime()
        actual_isi_duration = isi_offset - isi_onset

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
            fixation_cross.draw()
            win.flip()
            while global_clock.getTime() < mask_onset + mask_duration:
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

        # ------ LOCATION catch ------
        if image_data["location_catch"][i]:

            if el_tracker is not None:
                el_tracker.sendMessage("LOC_ONSET")

            key_to_loc = {"left": "L", "right": "R"}

            event.clearEvents(eventType='keyboard')
            fixation_arrows.draw()
            resp_rect_left.draw()
            resp_rect_right.draw()
            dot_left.draw()
            dot_right.draw()
            t_resp_onset  = win.flip()
            response_clock = core.Clock()

            # ── gaze response ─────────────────────────────────────────────────
            if gaze_response and el_tracker is not None:

                while response_loc is None and response_clock.getTime() < loc_response:
                    if "escape" in event.getKeys(keyList=["escape"]):
                        win.close(); core.quit()

                    try:
                        sample = el_tracker.getNewestSample()
                    except Exception:
                        sample = None

                    in_left,  _, _ = check_eye_in_location(
                        el_tracker, left_el[0],  left_el[1],
                        eye_used, fix_window_deg, ppd, sample=sample)
                    in_right, _, _ = check_eye_in_location(
                        el_tracker, right_el[0], right_el[1],
                        eye_used, fix_window_deg, ppd, sample=sample)

                    if in_left:
                        rt_loc       = response_clock.getTime()
                        response_loc = "L"
                    elif in_right:
                        rt_loc       = response_clock.getTime()
                        response_loc = "R"

                    if response_loc is not None:
                        win.flip(); break

            # ── keyboard response ─────────────────────────────────────────────
            else:
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
                    resp_rect_left.draw()
                    resp_rect_right.draw()
                    dot_left.draw()
                    dot_right.draw()
                    win.flip()

            correct_loc = response_loc == image_data["target_loc"][i] if response_loc else False
            fixation_cross.draw()
            win.flip()
            core.wait(postresp_fix)

        # ------ IDENTITY catch ------
        if image_data["identity_catch"][i]:
            fixation_cross.draw()
            win.flip()
            core.wait(preresp_fix)

            if el_tracker is not None:
                el_tracker.sendMessage("ID_ONSET")

            # stimuli
            target = image_data['only_targets'][target_id]

            distractor_match = image_data['only_targets'][int(image_data["distractor_selection_match_id"][i])]
            distractor_2 = image_data['only_targets'][int(image_data["distractor_selection_2_id"][i])]
            distractor_2_match = image_data['only_targets'][int(image_data["distractor_selection_2_match_id"][i])]
            
            # positions
            target.pos = loc_to_pos[image_data["target_selection_loc"][i]]

            distractor_match.pos = loc_to_pos[image_data["distractor_match_selection_loc"][i]]
            distractor_2.pos = loc_to_pos[image_data["distractor_2_selection_loc"][i]]
            distractor_2_match.pos = loc_to_pos[image_data["distractor_2_match_selection_loc"][i]]

            # sum-up
            response_stimuli = [target, distractor_match,distractor_2, distractor_2_match]
            
            # response mapping
            key_to_stim = {
                image_data["target_selection_loc"][i]: target,
                image_data["distractor_match_selection_loc"][i]: distractor_match,
                image_data["distractor_2_selection_loc"][i]: distractor_2,
                image_data["distractor_2_match_selection_loc"][i]: distractor_2_match}
            
            # draw stimuli and prep for keyboard response 
            event.clearEvents(eventType='keyboard')
            for stim in response_stimuli:
                stim.draw()

            t_resp_onset = win.flip()
            response_clock = core.Clock()

            # ── keyboard response ─────────────────────────────────────────────
            while response_id is None and response_clock.getTime() < id_response:
                keys = event.getKeys(keyList=["up", "down", "left", "right", "escape"], 
                                     timeStamped=response_clock)
                for key, t in keys:
                    if key == "escape":
                        win.close(); core.quit()
                    if key in key_to_stim:
                        rt_id       = t
                        response_id = key_to_stim[key].image
                        break
                if response_id is not None:
                    win.flip(); break
                
                for stim in response_stimuli:
                    stim.draw()
                win.flip()

            correct_id = response_id == image_data["target"][i] if response_id else False
            win.flip()
            core.wait(postresp_fix)

        # =====================================================
        # PRACTICE FEEDBACK
        # =====================================================
        if practice:
            if image_data["location_catch"][i]:
                fb = visual.TextStim(win, pos=(0, 0), height=40,
                    text=f"Location: {'Correct' if correct_loc else 'Incorrect'}",
                    color="green" if correct_loc else "red")
                fb.draw(); win.flip(); core.wait(2.0)
                
            if image_data["identity_catch"][i]:
                fb = visual.TextStim(win, pos=(-2, 0), height=40,
                    text=f"Identity: {'Correct' if correct_id else 'Incorrect'}",
                    color="green" if correct_id else "red")
                fb.draw(); win.flip(); core.wait(2.0)

        # ── trial end marker ──────────────────────────────────────────────────
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

        id_trials  = [t["correct_id"]  is True for t in trial_log if t["identity_catch"]]
        loc_trials = [t["correct_loc"] is True for t in trial_log if t["location_catch"]]
        acc_loc = np.mean(loc_trials) * 100 if loc_trials else 0
        acc_id = np.mean(id_trials) * 100 if id_trials else 0

    # =====================================================
    # END OF BLOCK SUMMARY
    # =====================================================
    if el_tracker is not None:
        el_tracker.stopRecording()
        el_tracker.closeDataFile()
        win.flip()
        local_edf = os.path.join(session_folder, edf_fname + ".EDF")
        el_tracker.receiveDataFile(edf_fname + ".EDF", local_edf)
        print(f"EDF saved to: {local_edf}")

    label = "Practice complete!" if practice else f"Block {run_num + 1} complete!"
    summary_text = visual.TextStim(
        win,
        text=f"{label}\n\nLocation: {acc_loc:.1f}%\n\nIdentity: {acc_id:.1f}%\n\nDo NOT press any button!\n\nThe experimenter will be with you soon.",
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