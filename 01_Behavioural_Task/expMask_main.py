from psychopy import visual, core, event, gui
from expMask_helpers import * 

# =====================================================
# Participant + Block info
# =====================================================
while True:
    myDlg = gui.Dlg(title="EXPMask Experiment")
    myDlg.addText('Subject Information')
    myDlg.addField("Subject ID:")
    myDlg.addField("Block ID:")
    myDlg.addField("Practice?", choices=["No", "Yes"])  # default = No
    ok_data = myDlg.show()

    if not myDlg.OK:
        print('user cancelled')
        quit()

    participant_num_str = ok_data[0]
    run_num_str = ok_data[1]
    practice_ = ok_data[2] == "Yes"   # False by default

    try:
        participant_num = int(participant_num_str)
        run_num = int(run_num_str)
    except ValueError:
        gui.popupError("Subject ID and Block ID must be integers.")
        continue

    output_dir = os.path.join(
        os.getcwd(),
        'data_expMask',
        f"sub-{participant_num:04d}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"sub-{participant_num:02d}_run-{run_num:02d}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # CHECK IF FILE EXISTS 
    if os.path.exists(output_path):
        overwriteDlg = gui.Dlg(title="File already exists")
        overwriteDlg.addText(
            f"The file:\n\n{output_filename}\n\nalready exists.\n\nOverwrite?"
        )
        overwriteDlg.addField("Overwrite file?", choices=["Yes", "No"])
        overwrite = overwriteDlg.show()

        if overwriteDlg.OK and overwrite[0] == "Yes":
            break  # proceed with experiment
        else:
            continue  # restart subject/run entry

    else:
        break  # file does not exist → proceed
        
# =====================================================
# WINDOW SETUP (must come first)
# =====================================================
win = visual.Window(
    size=(1024, 768),
    fullscr=True,
    screen=0,
    units='height',
    color=[0, 0, 0],
    colorSpace='rgb',
    waitBlanking=True
)

win.recordFrameIntervals = True  # timing diagnostics

# =====================================================
# CUE DATA
# =====================================================
cue_data = create_cue_dynam()
cue_data_practice = create_cue_dynam(highProb=0.8, lowProb=0.2, neutral=1.0, trials_per_cue=5)

# =====================================================
# IMAGE DATA
# =====================================================
stim_path = ".\\stimuli"
random_seed = participant_num + run_num 
image_data, stimuli = create_block_trials(stim_path, cue_data, random_seed=random_seed)
image_data_practice, stimuli_practice = create_block_trials(stim_path, cue_data_practice, random_seed=2026)
n_trials = len(image_data)
n_trial_practice = len(image_data_practice)

# =====================================================
# MASK SETUP
# =====================================================
masks_path = ".\\masks"
all_masks = os.listdir(masks_path)

# =====================================================
# POSITIONS AND SIZES
# =====================================================
ecc = 350 
pos_left = (-ecc, 0)
pos_right = ( ecc, 0)

image_size = 400
target_img_size = 150
eight_image_layout = False

# =====================================================
# STIMULI
# =====================================================
## ==== Create Cue Stim ==== ##
cue_stim = visual.TextStim(
    win, text='', height=0.08, ori=0.0, pos=(0, 0), colorSpace='rgb'
)

## ==== Create Fixation Cross ==== ##
fixation_cross= visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(50, 50), units='pix',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)

## ==== Create Target/Distractor Stim ==== ##
target_stim = visual.ImageStim(win,size=(image_size, image_size), units='pix')
distractor_stim = visual.ImageStim(win,size=(image_size, image_size), units='pix')

## ==== Mask Stim ==== ##
mask_left = visual.ImageStim(win, size=(image_size, image_size), units='pix', pos=pos_left)
mask_right = visual.ImageStim(win, size=(image_size, image_size), units='pix', pos=pos_right)

## ==== Create Arrows for selection ==== ##
arrow_left = visual.ImageStim(win, image=".\\arrows\\left.png", size=(target_img_size, target_img_size), units='pix')
arrow_right = visual.ImageStim(win, image=".\\arrows\\right.png", size=(target_img_size, target_img_size), units='pix')
arrow_up = visual.ImageStim(win, image=".\\arrows\\up.png", size=(target_img_size, target_img_size), units='pix')
arrow_down = visual.ImageStim(win, image=".\\arrows\\down.png", size=(target_img_size, target_img_size), units='pix')

# =====================================================
# DURATIONS
# =====================================================
cue_duration = 0.8
fix_duration = 0.8
image_duration = 0.017
loc_response = 1.5
id_response = 4

# =====================================================
# PRE-LOAD MASK POOL
# =====================================================
# Create a list of stimulus objects for every mask file
print("Loading masks into memory...")
mask_pool = []
for m_file in all_masks:
    full_path = os.path.join(masks_path, m_file)
    # We create the objects once here
    s = visual.ImageStim(win, image=full_path, size=(image_size, image_size), units='pix')
    mask_pool.append(s)
n_masks_per_trial = 12

# =====================================================
# PRE-LOAD Image POOL
# =====================================================
print("Loading stims into memory...")
image_data["stim"] = None  # pre-create column
image_data_practice["stim"] = None 
for i, img_path in enumerate(stimuli):
    stim = visual.ImageStim(
        win,
        image=img_path,
        size=(image_size, image_size),
        units="pix"
    )
    image_data.at[i, "stim"] = stim
    image_data_practice.at[i, "stim"] = stim

# =====================================================
# RUN REAL BLOCK
# =====================================================
if practice_:
    run_block(
    win,
    image_data_practice,
    stimuli_practice,
    cue_stim,
    fixation_cross,
    arrow_left,
    arrow_right,
    arrow_up,
    arrow_down,
    mask_pool,
    pos_left,
    pos_right,
    target_img_size,
    cue_duration,
    image_duration,
    loc_response,
    id_response,
    n_masks_per_trial,
    participant_num,
    run_num,
    output_dir,
    output_filename,
    practice=True)
    
else:
    run_block(
        win,
        image_data,
        stimuli,
        cue_stim,
        fixation_cross,
        arrow_left,
        arrow_right,
        arrow_up,
        arrow_down,
        mask_pool,
        pos_left,
        pos_right,
        target_img_size,
        cue_duration,
        image_duration,
        loc_response,
        id_response,
        n_masks_per_trial,
        participant_num,
        run_num,
        output_dir,
        output_filename,
        practice=False)