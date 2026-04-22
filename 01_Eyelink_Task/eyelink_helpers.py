import os
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, event, core


def setup_eyelink(win, edf_filename, edf_folder, dummy=False, calibration_required=True):
    """
    Connect to EyeLink, configure, open EDF, optionally calibrate.

    Parameters
    ----------
    win                  : psychopy Window (already created)
    edf_filename         : str, max 8 chars, no extension  e.g. 'loc0101'
    edf_folder           : str, where the EDF will be saved after transfer
    dummy                : bool, True = no physical tracker needed
    calibration_required : bool, run HV9 calibration on startup

    Returns
    -------
    el_tracker : pylink.EyeLink instance
    """

    # ── 1. Connect ────────────────────────────────────────────────────────────
    if dummy:
        el_tracker = pylink.EyeLink(None)
    else:
        try:
            el_tracker = pylink.EyeLink("100.1.1.1")
        except RuntimeError as e:
            print(f"EyeLink connection failed: {e}")
            core.quit()

    # ── 2. Open EDF on tracker PC ─────────────────────────────────────────────
    edf_file = edf_filename + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as e:
        print(f"Could not open EDF file: {e}")
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()

    preamble = f"RECORDED BY {os.path.basename(__file__)}"
    el_tracker.sendCommand(f"add_file_preamble_text '{preamble}'")

    # ── 3. Tracker version (determines available sample flags) ────────────────
    el_tracker.setOfflineMode()
    eyelink_ver = 0
    if not dummy:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print(f"EyeLink version: {vstr}  (ver {eyelink_ver})")

    # ── 4. Screen coordinates ─────────────────────────────────────────────────
    scn_w, scn_h = win.size
    el_tracker.sendCommand(f"screen_pixel_coords = 0 0 {scn_w-1} {scn_h-1}")
    el_tracker.sendMessage(f"DISPLAY_COORDS 0 0 {scn_w-1} {scn_h-1}")

    # ── 5. Data flags (version-aware, matching your original main script) ─────
    file_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT"
    link_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT"

    if eyelink_ver > 3:
        file_sample_flags = "LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT"
        link_sample_flags = "LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT"
    else:
        file_sample_flags = "LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT"
        link_sample_flags = "LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT"

    el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
    el_tracker.sendCommand(f"file_sample_data  = {file_sample_flags}")
    el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
    el_tracker.sendCommand(f"link_sample_data  = {link_sample_flags}")

    # ── 6. Calibration settings ───────────────────────────────────────────────
    el_tracker.sendCommand("calibration_type = HV9")
    el_tracker.sendCommand("calibration_area_proportion = 0.88 0.83")
    el_tracker.sendCommand("validation_area_proportion  = 0.88 0.83")

    # ── 7. Graphics environment ───────────────────────────────────────────────
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    pylink.openGraphicsEx(genv)

    # ── 8. Calibration ────────────────────────────────────────────────────────
    if calibration_required and not dummy:
        instruction = visual.TextStim(
            win,
            text=(
                "We will now calibrate the eye tracker.\n\n"
                "Please follow the dot and keep your gaze fixed on it.\n\n"
                "Press ENTER when ready."
            ),
            height=32, color=[-1, -1, -1], units="pix", wrapWidth=900)
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=["return"])
        el_tracker.doTrackerSetup()

    return el_tracker


def drift_check(el_tracker, win, fixation_cross, dummy=False):
    if dummy:
        return
    fixation_cross.draw()
    win.flip()
    scn_w, scn_h = win.size
    try:
        el_tracker.doDriftCorrect(scn_w // 2, scn_h // 2, 1, 1)
    except RuntimeError:
        el_tracker.doTrackerSetup()
    
    # doDriftCorrect stops recording — restart it
    el_tracker.startRecording(1, 1, 1, 1)
    pylink.msecDelay(50)


def close_eyelink(el_tracker, edf_filename, edf_folder):
    """
    Stop recording, close EDF, transfer file from tracker PC to local disk.
    Safe to call even if recording was never started.
    """
    try:
        if el_tracker.isRecording() == pylink.TRIAL_OK:
            el_tracker.stopRecording()
    except Exception:
        pass

    el_tracker.closeDataFile()

    local_path = os.path.join(edf_folder, edf_filename + ".EDF")
    try:
        el_tracker.receiveDataFile(edf_filename + ".EDF", local_path)
        print(f"EDF saved to: {local_path}")
    except RuntimeError as e:
        print(f"EDF transfer failed (dummy mode?): {e}")

    el_tracker.close()