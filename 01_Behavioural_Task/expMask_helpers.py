import numpy as np
import os
import pandas as pd 
from psychopy import visual, core, event, gui
from instructions import (
    INTRO_TEXT,
    TASK_TEXT,
    BREAK_TEXT,
    START_TEXT,
    PRACTICE_TEXT
)


def create_cue_dynam(highProb=0.7, lowProb=0.3, neutral=1.0, trials_per_cue=40):
    cue_data = {"cue_names": ["Sea animal",  "Water vessel",  "Neutral"],
                "cue_letters": ["SA", "WV", "X"],
                "cue_color": [[0.4, 0.6, 1.0], [-0.4, -0.2, 0.4], [-1, -1, -1]],
                "cue_highProb_cats": [["dolphin", "whale"], ["speedboat", "submarine"], 
                                    ["dolphin", "whale", "speedboat", "submarine"]],
                
                "cue_lowProb_cats": [["speedboat", "submarine"], ["dolphin", "whale"],
                                    ["dolphin", "whale", "speedboat", "submarine"]]}

    cue_data["cue_highProb"] = []
    cue_data["cue_lowProb"] = []
    
    for cue_id, cue in enumerate(cue_data["cue_names"]):
        if cue != "Neutral":
            cue_data["cue_highProb"].append([np.round(highProb / len(cue_data["cue_highProb_cats"][cue_id]), 2)] * len(cue_data["cue_highProb_cats"][cue_id]))
            cue_data["cue_lowProb"].append([np.round(lowProb / len(cue_data["cue_lowProb_cats"][cue_id]), 2)] * len(cue_data["cue_lowProb_cats"][cue_id]))
        
        else:
            cue_data["cue_highProb"].append([np.round(neutral / len(cue_data["cue_highProb_cats"][cue_id]), 3)] * len(cue_data["cue_highProb_cats"][cue_id]))
            cue_data["cue_lowProb"].append([np.round(neutral / len(cue_data["cue_lowProb_cats"][cue_id]), 3)] * len(cue_data["cue_highProb_cats"][cue_id]))
            
    cue_data["high_prob_trials"] = [np.array(x) * trials_per_cue for x in cue_data["cue_highProb"]] 
    cue_data["low_prob_trials"] =  [np.array(x) * trials_per_cue for x in cue_data["cue_lowProb"]]
    
    return cue_data
    
def build_constrained_order(df, seed=None, max_unexpected_run=1):
    rng = np.random.default_rng(seed)

    remaining = df.copy()
    ordered_rows = []

    last_target = None
    unexpected_run = 0

    while len(remaining) > 0:

        # valid candidates mask
        valid_mask = np.ones(len(remaining), dtype=bool)

        # Rule 1 — no same target twice
        if last_target is not None:
            valid_mask &= (remaining["target"].values != last_target)

        # Rule 2 — max unexpected run
        if unexpected_run >= max_unexpected_run:
            valid_mask &= (remaining["expectation"].values != "unexpected")

        valid = remaining[valid_mask]

        # if dead end → restart whole sequence
        if len(valid) == 0:
            return build_constrained_order(df, seed=rng.integers(0,1e9))

        # pick random valid row
        choice_idx = rng.integers(len(valid))
        row = valid.iloc[choice_idx]

        ordered_rows.append(row)

        # update state
        last_target = row["target"]
        if row["expectation"] == "unexpected":
            unexpected_run += 1
        else:
            unexpected_run = 0

        # remove selected row
        remaining = remaining.drop(valid.index[choice_idx])

    return pd.DataFrame(ordered_rows).reset_index(drop=True)

def create_block_trials(stim_path, cue_data, random_seed, long_isi=0.1): 
    categories = os.listdir(stim_path)
    stimuli = []
    for cat in categories:
        cat_path = os.path.join(stim_path, cat)
        files = os.listdir(cat_path)
        stimuli.extend([f".\\stimuli\\{cat}\\{x}" for x in files])

    stimuli = np.array(stimuli)[np.argsort(stimuli)]
    data = {"target_id": [],
            "distractor_id": [],
            "target": [],
            "distractor": [],
            "expectation": [],
            "mask_ISI": [],
            "cue": [],
            "cue_color": [],
            "cue_letter": [],
            "target_name": [],
            "target_cat": [],}

    mask_type = [0.017, long_isi]
    distractors = stimuli[np.char.count(stimuli, "mask") > 0]
    distractor_ids = np.arange(len(stimuli))[np.isin(stimuli, distractors)]
    local_distractor_ids = np.arange(0, len(distractors))

    for cue_id, cue in enumerate(cue_data["cue_names"]):
        high_cats = np.array(cue_data["cue_highProb_cats"][cue_id])
        low_cats = np.array(cue_data["cue_lowProb_cats"][cue_id])
        
        if cue != "Neutral":
            for i, l_cat in enumerate(low_cats):
                l_cat_stim = stimuli[np.char.count(stimuli, l_cat) > 0]
                targets = l_cat_stim[np.char.count(l_cat_stim, "mask") == 0]
                target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]
            

                trial_collectos = []
                for mask in mask_type:
                
                    h_trials = cue_data["low_prob_trials"][cue_id][i]
                    split_h_trials = int(h_trials // 2)
                    data["target_id"].extend(np.repeat(target_ids , split_h_trials))
                    data["target"].extend(np.repeat(targets, split_h_trials))
                    
                    target_names =[x.split("\\")[-1] for x in targets]
                    target_categories = [x.split("_")[0] for x in target_names]
                    
                    random_distractors = np.random.choice(local_distractor_ids, split_h_trials * 2)
                    data["distractor_id"].extend([distractor_ids[x] for x in random_distractors])
                    data["distractor"].extend([distractors[x] for x in random_distractors])
                    data["target_name"].extend(np.repeat(target_names , split_h_trials))
                    data["target_cat"].extend(np.repeat(target_categories , split_h_trials))
                    
                    if cue != "Neutral":
                        data["expectation"].extend(["unexpected"] * split_h_trials * 2)
                    else:
                        data["expectation"].extend(["neutral"]* split_h_trials * 2)
                        
                    data["mask_ISI"].extend([mask] * split_h_trials * 2)
                    data["cue"].extend([cue] * split_h_trials * 2)
                    data["cue_color"].extend([cue_data["cue_color"][cue_id]]* split_h_trials * 2)
                    data["cue_letter"].extend([cue_data["cue_letters"][cue_id]]* split_h_trials * 2)
                    
        for i, h_cat in enumerate(high_cats):
            h_cat_stim = stimuli[np.char.count(stimuli, h_cat) > 0]
            targets = h_cat_stim[np.char.count(h_cat_stim, "mask") == 0]
            target_ids = np.arange(len(stimuli))[np.isin(stimuli, targets)]

            trial_collectos = []
            for mask in mask_type:
                h_trials = cue_data["high_prob_trials"][cue_id][i]
                split_h_trials = int(h_trials // 2)
                
                data["target_id"].extend(np.repeat(target_ids , split_h_trials))
                data["target"].extend(np.repeat(targets , split_h_trials))
                
                target_names =[x.split("\\")[-1] for x in targets]
                target_categories = [x.split("_")[0] for x in target_names]
                
                random_distractors = np.random.choice(local_distractor_ids, split_h_trials * 2)
                data["distractor_id"].extend([distractor_ids[x] for x in random_distractors])
                data["distractor"].extend([distractors[x] for x in random_distractors])
                data["target_name"].extend(np.repeat(target_names , split_h_trials))
                data["target_cat"].extend(np.repeat(target_categories , split_h_trials))
                
                
                if cue != "Neutral":
                    data["expectation"].extend(["expected"] * split_h_trials * 2)
                else:
                    data["expectation"].extend(["neutral"]* split_h_trials * 2)
                    
                data["mask_ISI"].extend([mask] * split_h_trials * 2)
                data["cue"].extend([cue] * split_h_trials * 2)
                data["cue_color"].extend([cue_data["cue_color"][cue_id]]* split_h_trials * 2)
                data["cue_letter"].extend([cue_data["cue_letters"][cue_id]] * split_h_trials * 2)
                

                
    data["target_loc"] = [np.random.choice(["L", "R"], 1, p=[0.5, 0.5])[0] for _ in range(len(data["target"]))]
    data["distractor_loc"] = ["R" if x == "L" else "L" for x in data["target_loc"]]
    df = pd.DataFrame(data)
    df = build_constrained_order(df, seed=random_seed)
    return df, stimuli


def make_text_stim(win, text):
    return visual.TextBox2(
        win,
        text=text,
        letterHeight=0.03,    # 👈 instead of height
        pos=(0, 0),            # center of screen
        anchor='center',       # anchor box at its center
        alignment='center',    # center text inside the box
        size=(1.1, 0.8),
        color=[-1, -1, -1],
        font='Segoe UI Emoji'          # 👈 generally emoji-safe (OS dependent)
    )
    
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

    # geometry
    pos_left,
    pos_right,
    target_img_size, 

    # timing
    cue_duration,
    image_duration,
    loc_response,
    id_response,
    n_masks_per_trial,

    # bookkeeping
    participant_num,
    run_num,
    output_dir,
    output_filename,
    *,  
    practice=False):
        
    
    ## ==== Create all Texts ==== ##
    intro_text = make_text_stim(win, INTRO_TEXT)
    task_text = make_text_stim(win, TASK_TEXT)
    break_text = make_text_stim(win, BREAK_TEXT)
    start_text = make_text_stim(win, START_TEXT)
    stimuli_title = visual.TextStim(
    win,
    text="The stimuli you will see",
    height=0.06,
    pos=(0, 0.35),
    color=[-1, -1, -1]
    )

    stimuli_image = visual.ImageStim(
        win,
        image="instructions_stimuli.png",
        size=(1.0, 0.6),   # adjust if needed
        pos=(0, -0.05)
    )

    selection_options = 4
    
    print(f"Preparing params for identity selection:{1/selection_options} ..")
    only_target_indexes = image_data["target_id"].unique()
    only_targets = stimuli[only_target_indexes]
    # pre-create columns
    image_data["only_targets"] = None  
    image_data["only_targets_names"] = None

    for i, img_path in enumerate(stimuli):
        if img_path in only_targets:
            stim = visual.ImageStim(
                win,
                image=img_path,
                size=(target_img_size, target_img_size),
                units="pix"
            )
            image_data.at[i, "only_targets"] = stim
            image_data.at[i, "only_targets_names"] = img_path

    n_targets = len(only_targets)

    # circular layout 
    radius = 300
    arrow_radius = 150

    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)

    positions = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
    arrow_positions = [(arrow_radius*np.cos(a), arrow_radius*np.sin(a)) for a in angles]

    # Create a empty list to store all trial data 
    trial_log = []
    n_trials=(len(image_data))
    
    ## ==== Introdiction windows ==== ##
    draw_and_wait(win,
    lambda: intro_text.draw())
    
    draw_and_wait(
    win,
    lambda: (
        stimuli_title.draw(),
        stimuli_image.draw()))
    
    draw_and_wait(
        win,
        lambda: task_text.draw())
    
    if practice:
        practice_text = make_text_stim(win, PRACTICE_TEXT)
        draw_and_wait(win, lambda: practice_text.draw())
        feedback_text = visual.TextStim(win, text="", height=0.03,
            color=[-1, -1, -1],
            wrapWidth=1.0)
        
    draw_and_wait(win, lambda: start_text.draw())

    ## ==== Set up the Clock ==== ##
    global_clock= core.Clock()
    experiment_start_time = global_clock.getTime()
    
    for i in range(n_trials):
        
        # ==== Block break every 100 trials ====
        if (i + 1) % 100 == 0 and i != n_trials - 1:

            event.clearEvents(eventType="keyboard")

            while True:
                break_text.draw()
                win.flip()

                keys = event.getKeys(keyList=["space", "escape"])
                if "space" in keys:
                    break
                if "escape" in keys:
                    win.close()
                    core.quit()

            # Optional polish: re-center before next trial
            fixation_cross.draw()
            win.flip()
            core.wait(0.5)
        
        ## ==== ITI (jittered) ==== ##
        iti_duration = np.random.uniform(1.0, 2.0)
        fixation_cross.draw()
        win.flip()
        core.wait(iti_duration)
        
        ## ==== Cue ==== ##
        cue_stim.text = image_data["cue"][i]
        cue_stim.color = image_data["cue_color"][i]
        cue_stim.draw()
        win.flip()
        core.wait(cue_duration)
        
        ## ==== Pick trial images ==== ##
        target_id = image_data["target_id"][i]
        distractor_id = image_data["distractor_id"][i]
        current_target = image_data['stim'][target_id]
        current_distractor = image_data['stim'][distractor_id]

        ## ==== Set Positions ==== ##
        current_target.pos = pos_left if image_data["target_loc"][i] == "L" else pos_right
        current_distractor.pos = pos_left if image_data["distractor_loc"][i] == "L" else pos_right
        
        ## ==== Set ISI ==== ##
        isi_duration = image_data["mask_ISI"][i]
        
        # Get images onset, show images, draw fixation cross, flip and wait
        image_onset = global_clock.getTime()
        current_target.draw()
        current_distractor.draw()
        fixation_cross.draw()
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
        trial_masks_left = np.random.choice(mask_pool, n_masks_per_trial, replace=False)
        trial_masks_right = np.random.choice(mask_pool, n_masks_per_trial, replace=False)

        mask_i = 0
        mask_durations = []
        for lm_stim, rm_stim in zip(trial_masks_left, trial_masks_right):
            # Set positions (if they aren't already set)
            lm_stim.pos = pos_left
            rm_stim.pos = pos_right
            
            # Draw and Flip
            mask_onset = global_clock.getTime()
            lm_stim.draw()
            rm_stim.draw()
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
        
        ## ==== LOCATION RESPONSE WINDOW ==== ##
        event.clearEvents(eventType='keyboard')
        t_resp_onset = win.flip()
        response_clock = core.Clock()
        response_loc = None
        rt_loc = None

        while response_loc is None and response_clock.getTime() < loc_response:

            fixation_cross.draw()
            arrow_left.pos = pos_left
            arrow_right.pos = pos_right
            arrow_left.draw()
            arrow_right.draw()
            win.flip()

            keys = event.getKeys(
                keyList=['left', 'right', 'escape'],
                timeStamped=response_clock
            )

            for key, t in keys:
                if key == 'left':
                    response_loc = 'L'
                    rt_loc = t
                    break

                elif key == 'right':
                    response_loc = 'R'
                    rt_loc = t
                    break

                elif key == 'escape':
                    win.close()
                    core.quit()

        #print("Selected side:", str(response_loc), str(response_loc == image_data["target_loc"][i]))
        win.flip()
        core.wait(0.5)
        
        ## ==== IDENTITY RESPONSE WINDOW ==== ##
        target_selection = np.where(image_data["only_targets_names"] == image_data["target"][i])[0][0]

        distractors_selection = only_target_indexes[only_target_indexes != target_selection]
        wrong_choices = np.random.choice(image_data["only_targets"].iloc[distractors_selection], size=3, replace=False)
        stims = np.append(wrong_choices, image_data["only_targets"].iloc[target_selection])
        np.random.shuffle(stims)
        
        start_idx = np.random.randint(selection_options)
        event.clearEvents(eventType='keyboard')
        arrow_right.pos = arrow_positions[0]
        arrow_up.pos = arrow_positions[1]
        arrow_left.pos = arrow_positions[2]
        arrow_down.pos = arrow_positions[3]
        
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

        response_id = None
        rt_id = None
        key_to_index = {"right": 0, "up": 1, "left": 2,"down": 3}
        
        while response_id is None and response_clock.getTime() < id_response:
            
            keys = event.getKeys(keyList=["left", "right", "up", "down", "escape"],  
            timeStamped=response_clock)
            
            for key, t in keys:

                if key == "escape":
                    win.close()
                    core.quit()

                if key in key_to_index:
                    idx = key_to_index[key]
                    rt_id = t
                    response_id = stims[idx].image
                    break

            if response_id is not None:
                win.flip()  # clear screen immediately
                break

            # --- DRAW FRAME ---
            for stim, pos in zip(stims, positions):
                stim.pos = pos
                stim.draw()

            arrow_left.draw()
            arrow_right.draw()
            arrow_up.draw()
            arrow_down.draw()

            win.flip()

            
        #print("Selected image:", str(response_id), str(response_id == image_data["target"][i]))
        if practice:
            correct_loc = response_loc == image_data["target_loc"][i] if response_loc else False
            correct_id = response_id == image_data["target"][i] if response_id else False
            feedback_text.text = (
            f"Location: {'Correct' if correct_loc else 'Incorrect'}\n\n"
            f"Identity: {'Correct' if correct_id else 'Incorrect'}\n\n"
            "Press SPACE to continue")
            draw_and_wait(win, lambda: feedback_text.draw())

    
        #core.wait(1)
        
        ## ==== Logging ==== ##
        trial_log.append({
        "participant": participant_num,
        "block_number": run_num,
        "trial": i,
        "cue": image_data["cue"][i],
        "taget_name": image_data["target_name"][i],
        "tagte_cat": image_data["target_cat"][i],
        "target_loc": image_data["target_loc"][i],
        "image_onset": image_onset,
        "image_offset": image_offset, 
        "first_mask_onset": first_mask_onset,
        "first_mask_offset" : first_mask_offset,
        "last_mask_onset": mask_onset,
        "last_mask_offset": mask_offset,
        "isi_mask": isi_duration,
        "isi_mask_actual": actual_isi_duration,
        "expectation": image_data["expectation"][i],
        "response_loc": str(response_loc),
        "rt_loc": rt_loc if rt_loc else np.nan,
        "correct_loc": response_loc == image_data["target_loc"][i] if response_loc else 'None',
        "response_id": str(response_id),
        "rt_id": rt_id if rt_id else np.nan,
        "correct_id": response_id == image_data["target"][i] if response_id else 'None'})
        

        # ==== Save data ==== ##
        if practice == False:
            save_data(trial_log, output_dir, output_filename)
        
    loc_acc = np.mean([t["correct_loc"] is True for t in trial_log]) * 100
    id_acc = np.mean([t["correct_id"] is True for t in trial_log]) * 100

    if practice:
        summary_text = visual.TextStim(
        win,
        text=(
            f"Practice complete!\n\n"
            f"Location accuracy: {loc_acc:.1f}%\n"
            f"Identity accuracy: {id_acc:.1f}%\n\n"
            "Press Space to exit"
        ),
        height=0.05,
        color=[-1, -1, -1],
        wrapWidth=1.2)
    else:
        summary_text = visual.TextStim(
                win,
                text=(
                    f"Block {run_num + 1}/5 complete!\n\n"
                    f"Location accuracy: {loc_acc:.1f}%\n"
                    f"Identity accuracy: {id_acc:.1f}%\n\n"
                    "Press Space to exit"
                ),
                height=0.05,
                color=[-1, -1, -1],
                wrapWidth=1.2)

    draw_and_wait(win, lambda: summary_text.draw())    
    return trial_log

    
def save_data(data, output_dir, filename):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir,filename), index=False)