from psychopy import visual, core, event, gui, monitors
from expMask_helpers import * 

# =====================================================
# Paths
# =====================================================
stim_path = r".\stimuli"
masks_path = r".\masks"


# =====================================================
# DURATIONS
# =====================================================
cue_duration = 0.5
precue_fix = (0.8, 1.0)
postcue_fix = 0.5
image_duration = 0.017
id_response = 2
preresp_fix = 0.5
postresp_fix = 0.5 

# =====================================================
# POSITIONS AND SIZES  (now in pixels)
# =====================================================
image_size = 300        
cue_size = 150         
target_img_size = 200   
arrow_size = 50 
fix_size = 25
eight_image_layout = False

# =====================================================
# Participant + Block info
# =====================================================
while True:
    myDlg = gui.Dlg(title="EXPMask Experiment")
    myDlg.addText('Subject Information')
    myDlg.addField("Subject ID:")
    myDlg.addField("Block ID:")
    ok_data = myDlg.show()

    if not myDlg.OK:
        print('user cancelled')
        quit()

    participant_num_str = ok_data[0]
    run_num_str = ok_data[1]

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
    size=(1920, 1080),
    fullscr=True,
    screen=0,
    units='pix',
    color=[0, 0, 0],
    colorSpace='rgb',
    waitBlanking=True
)

win.recordFrameIntervals = True  # timing diagnostics

# =====================================================
# CUE DATA
# =====================================================
cue_data = create_cue_dynam(trials_per_cue=7)

# =====================================================
# IMAGE DATA
# =====================================================
random_seed = participant_num + run_num 
image_data, stimuli = create_block_trials(stim_path, cue_data, random_seed=random_seed, p=0.4, k=3, alpha=0.50)
n_trials = len(image_data)

# =====================================================
# MASK SETUP
# =====================================================
all_masks = os.listdir(masks_path)

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
# STIMULI
# =====================================================
## ==== Create Cue Stim ==== ##
cue_stim = visual.ImageStim(win,size=(cue_size, cue_size), units='pix', pos=(0, 0))

## ==== Create Fixation Cross ==== ##
fixation_cross = visual.ShapeStim(
    win=win, name='polygon', vertices='cross',
    size=(fix_size, fix_size),
    ori=0.0, pos=(0, 0), draggable=False, anchor='center',
    lineWidth=0.5,
    colorSpace='rgb', lineColor='black', fillColor='black',
    opacity=None, depth=0.0, interpolate=True)
        
## ==== Create Target Stim ==== ##
target_stim = visual.ImageStim(win,size=(image_size, image_size), units='pix', pos=(0, 0))
mask = visual.ImageStim(win,size=image_size, units='pix', pos=(0, 0))


arrow_left = visual.ImageStim(win, image=".\\arrows\\left.png", size=(arrow_size, arrow_size),  units='pix')
arrow_right = visual.ImageStim(win, image=".\\arrows\\right.png", size=(arrow_size, arrow_size),  units='pix')
arrow_up = visual.ImageStim(win, image=".\\arrows\\up.png", size=(arrow_size, arrow_size),  units='pix')
arrow_down = visual.ImageStim(win, image=".\\arrows\\down.png", size=(arrow_size, arrow_size),  units='pix')


# =====================================================
# PRE-LOAD Image POOL
# =====================================================
print("Loading stims into memory...")
image_data["stim"] = None 
for i, img_path in enumerate(stimuli):
    print(img_path)
    stim = visual.ImageStim(
        win,
        image=img_path,
        size=(image_size, image_size),
        units="pix"
    )

    image_data.at[i, "stim"] = stim

# =====================================================
# PreLoad Images for the selection 
# =====================================================
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


# =====================================================
# RUN REAL BLOCK
# =====================================================
run_block(
        win,
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
        
        participant_num,
        run_num,
        output_dir,
        output_filename,
        break_number=20,
        practice=True,
        send_trigger=None) # send_trigger = send_trigger when we want EEG
