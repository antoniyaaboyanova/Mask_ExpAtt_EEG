from psychopy import visual, core, event, gui, monitors
from expMask_helpers import *
from eyelink_helpers import setup_eyelink, drift_check, close_eyelink
import pylink
import time
import os, sys

# =====================================================
# Paths
# =====================================================
stim_path  = r".\stimuli"
masks_path = r".\masks"

# =====================================================
# DURATIONS
# =====================================================
cue_duration   = 0.5
precue_fix     = (0.8, 1.0)
postcue_fix    = 0.5
image_duration = 0.017
id_response    = 2
loc_response   = 2
preresp_fix    = 0.5
postresp_fix   = 0.5

# =====================================================
# POSITIONS AND SIZES  (pixels)
# =====================================================
ecc             = 350
pos_left        = (-ecc, 0)
pos_right       = ( ecc, 0)
image_size      = 300
cue_size        = 150
target_img_size = 200
arrow_size      = 50
fix_size        = 25

# =====================================================
# Participant + Block info
# =====================================================
while True:
    myDlg = gui.Dlg(title="EXPMask Experiment")
    myDlg.addText('Subject Information')
    myDlg.addField("Subject ID:")
    myDlg.addField("Block ID:")
    myDlg.addField("EyeLink:", choices=["True", "False"])
    myDlg.addField("Task Type:", choices=["loc", "ide"])
    ok_data = myDlg.show()

    if not myDlg.OK:
        print('user cancelled')
        quit()

    participant_num_str = ok_data[0]
    run_num_str         = ok_data[1]
    eyelink             = ok_data[2] == "True"
    task_type           = ok_data[3]

    try:
        participant_num = int(participant_num_str)
        run_num         = int(run_num_str)
    except ValueError:
        gui.popupError("Subject ID and Block ID must be integers.")
        continue

    output_dir = os.path.join(
        os.getcwd(), 'data_expMask', f"sub-{participant_num:04d}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"sub-{participant_num:02d}_run-{run_num:02d}_{task_type}.csv"
    output_path     = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        overwriteDlg = gui.Dlg(title="File already exists")
        overwriteDlg.addText(
            f"The file:\n\n{output_filename}\n\nalready exists.\n\nOverwrite?")
        overwriteDlg.addField("Overwrite file?", choices=["Yes", "No"])
        overwrite = overwriteDlg.show()
        if overwriteDlg.OK and overwrite[0] == "Yes":
            break
        else:
            continue
    else:
        break

# =====================================================
# EDF filename  (max 8 chars, no extension)
# =====================================================
edf_fname = f"{task_type}{run_num:02d}{participant_num:02d}"  # e.g. "loc0101"
eyetracking_folder = 'eyelink_results'
session_folder = os.path.join(eyetracking_folder, f"sub-{participant_num:04d}")
os.makedirs(session_folder, exist_ok=True)

# =====================================================
# WINDOW  (single, created once)
# =====================================================
mon = monitors.Monitor('myMonitor', width=41.0, distance=57.0)
mon.setSizePix((1280, 1024))
win = visual.Window(
    size=(1280, 1024),
    fullscr=True,
    monitor=mon,
    screen=0,
    units='pix',
    color=[0, 0, 0],
    colorSpace='rgb',
    waitBlanking=True)
win.recordFrameIntervals = True

# =====================================================
# EYELINK SETUP  (after window — genv needs win)
# =====================================================
dummy_mode = not eyelink
el_tracker = setup_eyelink(
    win=win,
    edf_filename=edf_fname,
    edf_folder=session_folder,
    dummy=dummy_mode,
    calibration_required=True)

# =====================================================
# CUE + IMAGE DATA
# =====================================================
cue_data = create_cue_dynam()

random_seed = participant_num + run_num
if task_type == "ide":
    identity_catch_prob = 1.0
    location_catch_prob = 0.0
else:
    identity_catch_prob = 0.0
    location_catch_prob = 1.0

image_data, stimuli = create_block_trials(
    stim_path, cue_data,
    random_seed=random_seed,
    identity_catch=identity_catch_prob,
    location_catch=location_catch_prob)

# =====================================================
# STIMULI
# =====================================================
cue_stim = visual.ImageStim(win, size=(cue_size, cue_size), units='pix', pos=(0, 0))

fixation_cross = visual.ShapeStim(
    win=win, vertices='cross',
    size=(fix_size, fix_size),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=0.5, colorSpace='rgb',
    lineColor='black', fillColor='black',
    interpolate=True)

fixation_arrows = visual.TextStim(
    win=win, text='<<  >>',
    font='Arial', units='pix',
    pos=(0, 0), height=40, color='black')

target_stim     = visual.ImageStim(win, size=(image_size, image_size), units='pix')
distractor_stim = visual.ImageStim(win, size=(image_size, image_size), units='pix')
mask_left       = visual.ImageStim(win, size=(image_size, image_size), units='pix', pos=pos_left)
mask_right      = visual.ImageStim(win, size=(image_size, image_size), units='pix', pos=pos_right)

# =====================================================
# PRE-LOAD MASK POOL
# =====================================================
all_masks = os.listdir(masks_path)
print("Loading masks into memory...")
mask_pool = []
for m_file in all_masks:
    s = visual.ImageStim(win, image=os.path.join(masks_path, m_file),
                         size=(image_size, image_size), units='pix')
    mask_pool.append(s)
n_masks_per_trial = 12

# =====================================================
# PRE-LOAD IMAGE POOL
# =====================================================
print("Loading stims into memory...")
image_data["stim"] = None
for i, img_path in enumerate(stimuli):
    image_data.at[i, "stim"] = visual.ImageStim(
        win, image=img_path, size=(image_size, image_size), units="pix")

# =====================================================
# PRE-LOAD SELECTION IMAGES
# =====================================================
print("Preparing identity selection stims...")
only_target_indexes = image_data["target_id"].unique()
only_targets        = stimuli[only_target_indexes]

image_data["only_targets"]       = None
image_data["only_targets_names"] = None
for i, img_path in enumerate(stimuli):
    if img_path in only_targets:
        image_data.at[i, "only_targets"] = visual.ImageStim(
            win, image=img_path, size=(target_img_size, target_img_size), units="pix")
        image_data.at[i, "only_targets_names"] = img_path

# =====================================================
# RUN BLOCK
# =====================================================
try:
    print("Starting run_block...")
    run_block(
        win,
        el_tracker,
        image_data,
        stimuli,
        cue_stim,
        fixation_cross,
        fixation_arrows,
        mask_pool,
        pos_left,
        pos_right,
        
        cue_duration,
        precue_fix,
        postcue_fix,
        image_duration,
        id_response,
        loc_response,
        preresp_fix,
        postresp_fix,
        n_masks_per_trial,
        
        participant_num,
        run_num,
        output_dir,
        output_filename,
        session_folder,
        edf_fname,
        break_number=80,
        practice=False,
        dummy=dummy_mode,
        instruct=task_type) # ← NEW

# =====================================================
# CLEAN SHUTDOWN  (always runs, even on crash)
# =====================================================
finally:
    # Only reached on crash or escape — normal exit already transferred in run_block
    if eyelink and el_tracker.isConnected():
        try:
            el_tracker.stopRecording()
            el_tracker.closeDataFile()
            local_edf = os.path.join(session_folder, edf_fname + ".EDF")
            el_tracker.receiveDataFile(edf_fname + ".EDF", local_edf)
        except Exception as e:
            print(f"EDF transfer on abort failed: {e}")
        el_tracker.close()
    win.close()
    core.quit()