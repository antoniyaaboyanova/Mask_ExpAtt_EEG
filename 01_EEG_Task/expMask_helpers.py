import numpy as np
import random
import os
import pandas as pd 
from collections import Counter

from psychopy import visual, core, event, gui
from psychopy import parallel
from instructions import (
    INTRO_TEXT,
    TASK_TEXT,
    BREAK_TEXT,
    START_TEXT,
    PRACTICE_TEXT
)

# =====================================================
# Trial Functions
# =====================================================
def create_cue_dynam(highProb=0.7, lowProb=0.3, neutral=1.0, trials_per_cue=40):
    
    # double the amount for EEG 
    trial_per_neutral = 2 * trials_per_cue
    
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
    if row['mask_ISI'] == 0.017:
        mask_code = 10
    elif row['mask_ISI'] == late:
        mask_code = 20
    else:
        raise ValueError("Unknown mask type")
    
    # ---- IMAGE (side dependent) ----
    image_code = row['image_index']
    base_code = mask_code + image_code
    
    # ---- EXPECTATION ----
    if row['expectation'] == 'neutral':
        exp_code = 0
    elif row['expectation'] == 'expected':
        exp_code = 100
    elif row['expectation'] == 'unexpected':
        exp_code = 200
    else:
        raise ValueError("Unknown expectation")
    
    return base_code + exp_code
    
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

def assign_catch_trials(df, rng=None, p=0.25, k=3, alpha=0.64):
    """
    Takes your existing 320-trial dataframe and:
      1. Flags 80 rows as catch trials (52 neutral, 14 expected, 14 unexpected),
         balanced across mask_ISI within each expectation condition.
      2. Reorders the sequence so catches are spaced min=2, max=7-8, mean~4 apart.
    
    Adds an 'identity_catch' boolean column.
    Returns a reordered dataframe (reset index).
    """
    if rng is None:
        rng = random.Random()

    df = df.copy()
    df['identity_catch'] = False

    # --- 1. Flag catch trials ---
    neut, exp, unexp = allocate_catch_trials(len(df), p, k, alpha)
    catch_counts = {'neutral': neut, 'expected': exp, 'unexpected': unexp}

    for condition, n_catches in catch_counts.items():
        cond_idx = df[df['expectation'] == condition].index.tolist()
        assert len(cond_idx) >= n_catches, \
            f"Not enough {condition} trials: need {n_catches}, have {len(cond_idx)}"

        # Balance across mask_ISI: half from 0.017, half from 0.100
        half = n_catches // 2  # both 52 and 14 are even, so no remainder

        for isi_val, n in [(0.017, half), (0.100, half)]:
            isi_idx = df.loc[cond_idx][df.loc[cond_idx, 'mask_ISI'] == isi_val].index.tolist()
            assert len(isi_idx) >= n, \
                f"Not enough {condition}/ISI={isi_val} trials: need {n}, have {len(isi_idx)}"
            chosen = rng.sample(isi_idx, n)
            df.loc[chosen, 'identity_catch'] = True

    # --- 2. Separate catches and non-catches ---
    catch_df    = df[df['identity_catch']].sample(frac=1, random_state=rng.randint(0, 99999)).reset_index(drop=True)
    noncatch_df = df[~df['identity_catch']].sample(frac=1, random_state=rng.randint(0, 99999)).reset_index(drop=True)

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

def validate_sequence(df):
    catches = df[df['identity_catch']]
    print(f"Total:     {len(df)}")
    print(f"Catch:     {len(catches)}  ({len(catches)/len(df)*100:.1f}%)")
    print(f"Non-catch: {len(df) - len(catches)}")

    print(f"\nCatch breakdown by expectation:")
    for cond, grp in catches.groupby('expectation'):
        c017 = (grp['mask_ISI'] == 0.017).sum()
        c100 = (grp['mask_ISI'] == 0.100).sum()
        print(f"  {cond:11s}: total={len(grp)}  ISI=0.017: {c017}  ISI=0.100: {c100}")

    catch_pos = df.index[df['identity_catch']].tolist()
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

def create_block_trials(stim_path, cue_data, random_seed, long_isi=0.1, pick_images="_01", identity_catch=0.1, p=0.25, k=3, alpha=0.64): 
    
    rng = random.Random(random_seed) 
    categories = os.listdir(stim_path)
    stimuli = []
    for cat in categories:
        cat_path = os.path.join(stim_path, cat)
        files = os.listdir(cat_path)
        stimuli.extend([f".\\stimuli\\{cat}\\{x}" for x in files])

    # Filter stims
    stimuli = np.array(stimuli)
    stimuli = np.array([x for x in stimuli if pick_images in x ])
    stimuli = stimuli[np.argsort(stimuli)]
    

    data = {"target_id": [],
            "target": [],
            "expectation": [],
            "mask_ISI": [],
            "cue": [],
            "target_name": [],
            "target_cat": []}

    mask_type = [0.017, long_isi]
    for cue_id, cue in enumerate(cue_data["cue_names"]):
        high_cats = np.array(cue_data["cue_highProb_cats"][cue_id])
        low_cats = np.array(cue_data["cue_lowProb_cats"][cue_id])

        # since the neutral category has all 4 images there is no need to repeat it twice
        if cue != ".\\cues\\Neutral.png":
            # This loop handles only unexpexted cases
            for i, l_cat in enumerate(low_cats):
                l_cat_stim = stimuli[np.char.count(stimuli, l_cat) > 0]
                targets = l_cat_stim[np.char.count(l_cat_stim, "mask") == 0]
                target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]
    
                for mask in mask_type:
                
                    h_trials = int(cue_data["low_prob_trials"][cue_id][i])
                
                    data["target_id"].extend(np.repeat(target_ids , h_trials))
                    data["target"].extend(np.repeat(targets, h_trials))
                    
                    target_names =[x.split("\\")[-1] for x in targets]
                    target_categories = [x.split("_")[0] for x in target_names]               

                    data["target_name"].extend(np.repeat(target_names , h_trials))
                    data["target_cat"].extend(np.repeat(target_categories , h_trials))
                    data["expectation"].extend(["unexpected"] * h_trials)       
                    data["mask_ISI"].extend([mask] * h_trials)
                    data["cue"].extend([cue] * h_trials)
        
        # This loop handles expexted and neutral cases         
        for i, h_cat in enumerate(high_cats):
            h_cat_stim = stimuli[np.char.count(stimuli, h_cat) > 0]
            targets = h_cat_stim[np.char.count(h_cat_stim, "mask") == 0]
            target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]

            for mask in mask_type:
                h_trials = int(cue_data["high_prob_trials"][cue_id][i])
                
                data["target_id"].extend(np.repeat(target_ids , h_trials))
                data["target"].extend(np.repeat(targets , h_trials))
                
                target_names =[x.split("\\")[-1] for x in targets]
                target_categories = [x.split("_")[0] for x in target_names]
            
                data["target_name"].extend(np.repeat(target_names , h_trials))
                data["target_cat"].extend(np.repeat(target_categories , h_trials))
                
                if cue != ".\\cues\\Neutral.png":
                    data["expectation"].extend(["expected"] * h_trials)
                else:
                    data["expectation"].extend(["neutral"]* h_trials)
                    
                data["mask_ISI"].extend([mask] * h_trials)
                data["cue"].extend([cue] * h_trials)
            
    mapping = {0: 1,
               3: 2,
               1: 3,
               2: 4}
    
    df = pd.DataFrame(data)
    df['image_index'] = df['target_id'].map(mapping)
    df['trigger'] = df.apply(assign_trigger, axis=1)
    df = build_constrained_order(df, rng=rng)
    df = assign_catch_trials(df, rng=rng, p=p, k=k, alpha=alpha)
    validate_sequence(df)
    
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

port = parallel.ParallelPort(address=0x3FD8)  # LPT1
def send_trigger(code):
    port.setData(int(code))
    core.wait(0.005)   # 5 ms pulse
    port.setData(0)
    
def run_block(win,
    image_data,
    stimuli,
    
    # stimuli & layout
    cue_stim,
    fixation_cross,
    arrow_left,
    arrow_right,
    arrow_up,
    arrow_down,
    mask_pool,
    target_img_size,


    # timing
    cue_duration, 
    precue_fix, 
    postcue_fix, 
    image_duration, 
    id_response,
    preresp_fix,
    postresp_fix,
    n_masks_per_trial,

    # bookkeeping
    participant_num,
    run_num,
    output_dir,
    output_filename,
    *,
    break_number=80,
    practice=False,
    send_trigger=None):
        
    
    # =====================================================
    # TEXT CREATION
    # =====================================================
    intro_text = make_text_stim(win, INTRO_TEXT)
    task_text = make_text_stim(win, TASK_TEXT)
    start_text = make_text_stim(win, START_TEXT)
    stimuli_title = visual.TextStim(
        win,
        text="The stimuli you will see",
        height=40,          
        pos=(0, 400),       
        color=[-1, -1, -1],
        units="pix",        
        wrapWidth=856
    )
    stimuli_image = visual.ImageStim(
        win,
        image="instructions_stimuli.png",
        size=(800, 800),    
        pos=(0, 0),
        units='pix'        
    )
    cues_image = visual.ImageStim(
        win,
        image="instructions_cues.png",
        size=(800, 400),    
        pos=(0, 0),
        units='pix'        
    )
    cues_title = visual.TextStim(
        win,
        text="The cues you will see. The X symbolises neutrality, meaning it holds no predictive power of what will follow.",
        height=40,        
        pos=(0, 250),      
        color=[-1, -1, -1],
        units="pix",        
        wrapWidth=856
    )

    radius = 250            
    arrow_radius = 100      
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    positions = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
    arrow_positions = [(arrow_radius*np.cos(a), arrow_radius*np.sin(a)) for a in angles]

    # Create a empty list to store all trial data 
    trial_log = []
    block_log = []
    n_trials=(len(image_data))
    
    # =====================================================
    # INTRODUCTION
    # =====================================================
    draw_and_wait(win,
    lambda: intro_text.draw())
    
    draw_and_wait(
    win,
    lambda: (
        stimuli_title.draw(),
        stimuli_image.draw()))
    
    draw_and_wait(
    win,
    lambda: (
        cues_title.draw(),
        cues_image.draw()))
            
    
    
    draw_and_wait(
        win,
        lambda: task_text.draw())
    
    if practice:
        print(f"There will be {n_trials} trials in the practice")
        practice_text = make_text_stim(win, PRACTICE_TEXT)
        draw_and_wait(win, lambda: practice_text.draw())
        feedback_text = visual.TextStim(win, text="", height=2,
            color=[-1, -1, -1],
            wrapWidth=40)
        
    draw_and_wait(win, lambda: start_text.draw())

    ## ==== Set up the Clock ==== ##
    global_clock= core.Clock()
    experiment_start_time = global_clock.getTime()
    
    for i in range(n_trials):
        
        # =====================================================
        # INTERIM BREAK 
        # =====================================================
        if (i + 1) % break_number == 0 and i != n_trials - 1:
            print("Participant is having a break...")
            event.clearEvents(eventType="keyboard")

            id_trials = [
                t["correct_id"] is True
                for t in block_log       # ← was trial_log
                if t["identity_catch"]
            ]

            id_acc = np.mean(id_trials) * 100 if id_trials else 0

            break_text = visual.TextStim(
                win,
                text=(
                    f"Please take a short break (1–2 minutes).\n\n"
                    f"Identity accuracy: {id_acc:.1f}%\n\n"
                    "Press SPACE to continue ⌨"
                ),
                height=32,
                color=[-1, -1, -1],
                wrapWidth=856)

            draw_and_wait(win, lambda: break_text.draw())
            block_log = []

            fixation_cross.draw()
            win.flip()
            core.wait(0.5)
        
        # =====================================================
        # CUE PERIOD
        # =====================================================
        ## ==== ITI (jittered) ==== ##
        iti_duration = np.random.uniform(precue_fix[0], precue_fix[1])
        fixation_cross.draw()
        win.flip()
        core.wait(iti_duration)
        
        ## ==== Cue ==== ##
        cue_path = image_data["cue"][i]
        cue_stim.image = cue_path
        cue_stim.draw()
     
        cue_stim.draw()
        if send_trigger is not None:
            win.callOnFlip(send_trigger, 1)

        win.flip()
        core.wait(cue_duration)
        fixation_cross.draw()
        win.flip()
        core.wait(postcue_fix)
        
        # =====================================================
        # IMAGE + MASK PERIOD
        # =====================================================
        ## ==== Pick trial image ==== ##
        target_id = image_data["target_id"][i]
        current_target = image_data['stim'][target_id]

        ## ==== Set ISI ==== ##
        isi_duration = image_data["mask_ISI"][i]
        
        # Get images onset, show images, draw fixation cross, flip and wait
        image_onset = global_clock.getTime()
        current_target.draw()
        fixation_cross.draw()
        
        # EEG triggers 
        if send_trigger is not None:
            trig = int(image_data["trigger"][i])
            win.callOnFlip(send_trigger, trig)
        
        win.flip()
        
        while global_clock.getTime() < image_onset + image_duration:
            pass

        # Get images offset and measure actual image duration
        image_offset = global_clock.getTime()
        actual_image_duration = image_offset - image_onset
        print(f"Trial {i}: Image duration = {actual_image_duration}")

        # Get ISI onset, draw fixation cross and wait
        isi_onset = global_clock.getTime()
        fixation_cross.draw()
        win.flip()
        
        while global_clock.getTime() < isi_onset + isi_duration:
            pass
            
        # Get ISI offset and measure actual ISI duration
        isi_offset= global_clock.getTime()
        actual_isi_duration= isi_offset - isi_onset
        #print(f"Trial {i}: ISI duration = {actual_isi_duration}")
        
        ## ==== MASK SEQUENCE ==== ##
        # Randomly select stimulus objects from our pre-loaded pool
        trial_masks = np.random.choice(mask_pool, n_masks_per_trial, replace=False)

        mask_i = 0
        mask_durations = []
        for m_stim in trial_masks:

            # Draw and Flip
            mask_onset = global_clock.getTime()
            m_stim.draw()
            fixation_cross.draw()
            win.flip()
            
            while global_clock.getTime() < mask_onset + image_duration:
                pass
            
            # Get image offset and measure actual image duration
            mask_offset = global_clock.getTime()
            mask_durations.append(mask_offset - mask_onset)
            if mask_i == 0:
                first_mask_onset = mask_onset
                first_mask_offset = mask_offset
            mask_i += 1
        
        actual_mask_duration = np.mean(mask_durations)
        fixation_cross.draw()
        short_isi_st = win.flip()
            

        # =====================================================
        # IDENTITY RESPONSE PERIOD
        # =====================================================
        if image_data["identity_catch"][i]:
            fixation_cross.draw()
            win.flip()
            core.wait(preresp_fix)

            # Always show all 4 targets in fixed order
            stims = image_data["only_targets"].iloc[0:4].reset_index(drop=True)

            # Assign arrow positions to match circular layout indices
            arrow_right.pos = arrow_positions[0]
            arrow_up.pos    = arrow_positions[1]
            arrow_left.pos  = arrow_positions[2]
            arrow_down.pos  = arrow_positions[3]

            # Key → stim index must match the arrow/position assignment above
            key_to_index = {"right": 0, "up": 1, "left": 2, "down": 3}

            event.clearEvents(eventType='keyboard')
            response_id = None
            rt_id = None

            # Initial draw
            for stim, pos in zip(stims, positions):
                stim.pos = pos
                stim.draw()
            
            arrow_left.draw()
            arrow_right.draw()
            arrow_up.draw()
            arrow_down.draw()
            
            t_resp_onset = win.flip()
            response_clock = core.Clock()

            while response_id is None and response_clock.getTime() < id_response:
                keys = event.getKeys(
                    keyList=["left", "right", "up", "down", "escape"],
                    timeStamped=response_clock)

                for key, t in keys:
                    if key == "escape":
                        win.close()
                        core.quit()
                    
                    if key in key_to_index:
                        idx = key_to_index[key]
                        rt_id = t
                        response_id = stims.iloc[idx].image
                        break

                if response_id is not None:
                    win.flip()
                    break

                # Redraw frame
                for stim, pos in zip(stims, positions):
                    stim.pos = pos
                    stim.draw()
                
                arrow_left.draw()
                arrow_right.draw()
                arrow_up.draw()
                arrow_down.draw()
                win.flip()
                
            # after response loop
            fixation_cross.draw()
            win.flip()
            core.wait(postresp_fix)  # ← inside the if block, with fixation visible
        
        else:
            response_id = None
            rt_id = None
            
            
        # =====================================================
        # PRACTICE AFTER TRIAL FEEDBACK
        # =====================================================
        if practice:
            if image_data["identity_catch"][i]:
                
                feedback_id  = visual.TextStim(win, pos=(0, 0), height=40)
                correct_id = response_id == image_data["target"][i] if response_id else False
                feedback_id.text  = f"Identity: {'Correct' if correct_id else 'Incorrect'}"
                feedback_id.color  = "green" if correct_id else "red"
                feedback_id.draw()
                win.flip()
                core.wait(2.0)

                

        
        ## ==== Logging ==== ##
        trial_log.append({
        "participant": participant_num,
        "block_number": run_num,
        "trial": i,
        "trigger": image_data["trigger"][i],
        "cue": image_data["cue"][i],
        "taget_name": image_data["target_name"][i],
        "target_cat": image_data["target_cat"][i],
        "image_onset": image_onset,
        "image_offset": image_offset, 
        "first_mask_onset": first_mask_onset,
        "first_mask_offset" : first_mask_offset,
        "last_mask_onset": mask_onset,
        "last_mask_offset": mask_offset,
        "isi_mask": isi_duration,
        "isi_mask_actual": actual_isi_duration,
        "expectation": image_data["expectation"][i],
        "identity_catch": image_data["identity_catch"][i],
        "response_id": str(response_id),
        "rt_id": rt_id if rt_id else np.nan,
        "correct_id": response_id == image_data["target"][i] if response_id else 'None'})
        
        block_log.append(trial_log[-1])
        
        print("Selected image:", str(response_id), str(response_id == image_data["target"][i]))
        

        # ==== Save data ==== ##
        if practice == False:
            save_data(trial_log, output_dir, output_filename)
        
        id_trials = [
            t["correct_id"] is True
            for t in trial_log
            if t["identity_catch"]
        ]

        id_acc = np.mean(id_trials) * 100 if id_trials else 0

    if practice:
        summary_text = visual.TextStim(
        win,
        text=(
            f"Practice complete!\n\n"
            f"Identity accuracy: {id_acc:.1f}%\n\n"
            "Press Space to exit"
        ),
        height=40,
        color=[-1, -1, -1],
        wrapWidth=856)
    else:
        summary_text = visual.TextStim(
                win,
                text=(
                    f"Block {run_num + 1} complete!\n\n"
                    f"Identity accuracy: {id_acc:.1f}%\n\n"
                    "Press Space to exit"
                ),
                height=40,
                color=[-1, -1, -1],
                wrapWidth=856)

    draw_and_wait(win, lambda: summary_text.draw())    
    return trial_log

    
def save_data(data, output_dir, filename):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir,filename), index=False)