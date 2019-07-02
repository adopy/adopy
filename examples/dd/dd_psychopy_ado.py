import os
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import core, visual, event, data, gui

from adopy import Engine
from adopy.tasks.dd import TaskDD, ModelHyp

###############################################################################
# Global variables
###############################################################################

PATH_DATA = Path('./data')

BOX_W = 6
BOX_H = 6
DIST_BTWN = 8

LINE_WIDTH = 8
LINE_COLOR = '#ffffff'
COLOR_RED = '#ff0033'
COLOR_BLUE = '#0010ff'
COLOR_GRAY = '#939393'
COLOR_HIDDEN = '#555555'

TEXT_FONT = 'Arial'
TEXT_SIZE = 2
TEXT_MARGIN = 1

FIXATION_SIZE = 1

KEYS_LEFT = ['left', 'z', 'f']
KEYS_RIGHT = ['right', 'slash', 'j']

INSTRUCTION = [
    # 0
    """
This task is the delay discounting task.

On every trial, two options will be presented on the screen.

Each option has a possible reward you can earn and

a delay to obtain the reward.


Press <space> to proceed.
""",
    # 1
    """
You should choose what you prefer between two options

by pressing <f> (left option) or <j> (right option).


Press <space> to proceed.
""",
    # 2
    """
Let's do some practices to check if you understand the task.


Press <space> to start practices.
""",
    # 3
    """
Great job. Now, Let's get into the main task.

Press <space> to start a main game.
""",
    # 4
    """
You completed all the game.

Thanks for your participation.


Press <space> to end.
""",
]

###############################################################################
# Functions for the delay discounting task
###############################################################################


def convert_delay_to_str(delay):
    """Convert a delay value in a weekly unit into a human-readable string."""
    tbl_conv = {
        0: 'Now',
        0.43: 'In 3 days',
        0.714: 'In 5 days',
        1: 'In 1 week',
        2: 'In 2 weeks',
        3: 'In 3 weeks',
        4.3: 'In 1 month',
        6.44: 'In 6 weeks',
        8.6: 'In 2 months',
        10.8: 'In 10 weeks',
        12.9: 'In 3 months',
        17.2: 'In 4 months',
        21.5: 'In 5 months',
        26: 'In 6 months',
        52: 'In 1 year',
        104: 'In 2 years',
        156: 'In 3 years',
        260: 'In 5 years',
        520: 'In 10 years'
    }
    mv, ms = None, None
    for (v, s) in tbl_conv.items():
        if mv is None or np.square(delay - mv) > np.square(delay - v):
            mv, ms = v, s
    return ms


def show_instruction(caption):
    global window

    text = visual.TextStim(window, caption, font=TEXT_FONT,
                           pos=(0, 0), bold=True, height=0.7, wrapWidth=30)
    text.draw()
    window.flip()

    _ = event.waitKeys(keyList=['space'])


def show_countdown():
    global window

    text1 = visual.TextStim(window, text='1', pos=(0., 0.), height=2)
    text2 = visual.TextStim(window, text='2', pos=(0., 0.), height=2)
    text3 = visual.TextStim(window, text='3', pos=(0., 0.), height=2)

    text3.draw()
    window.flip()
    core.wait(1)

    text2.draw()
    window.flip()
    core.wait(1)

    text1.draw()
    window.flip()
    core.wait(1)


def draw_option(delay, reward, direction, chosen=False):
    """Draw an option with given delay and reward value."""
    global window

    pos_x_center = direction * DIST_BTWN
    pos_x_left = pos_x_center - BOX_W
    pos_x_right = pos_x_center + BOX_W
    pos_y_top = BOX_H / 2
    pos_y_bottom = -BOX_H / 2

    fill_color = 'darkgreen' if chosen else None

    # Show the option box
    box = visual.ShapeStim(window,
                           lineWidth=8,
                           lineColor='white',
                           fillColor=fill_color,
                           vertices=((pos_x_left, pos_y_top),
                                     (pos_x_right, pos_y_top),
                                     (pos_x_right, pos_y_bottom),
                                     (pos_x_left, pos_y_bottom)))
    box.draw()

    # Show the reward
    text_a = visual.TextStim(window,
                             '${:,.0f}'.format(reward),
                             font=TEXT_FONT,
                             pos=(pos_x_center, TEXT_MARGIN))
    text_a.size = TEXT_SIZE
    text_a.draw()

    # Show the delay
    text_d = visual.TextStim(window,
                             convert_delay_to_str(delay),
                             font=TEXT_FONT,
                             pos=(pos_x_center, -TEXT_MARGIN))
    text_d.size = TEXT_SIZE
    text_d.draw()


def run_trial(design):
    """Run one trial for the delay discounting task using PsychoPy."""
    global window

    # Direction: -1 (L - LL / R - SS) or
    #            1  (L - SS / R - LL)
    direction = np.random.randint(0, 2) * 2 - 1  # Return -1 or 1
    is_ll_on_left = int(direction == -1)

    draw_option(design['t_ss'], design['r_ss'], -1 * direction)
    draw_option(design['t_ll'], design['r_ll'], 1 * direction)
    window.flip()

    timer = core.Clock()
    keys = event.waitKeys(keyList=KEYS_LEFT + KEYS_RIGHT)
    rt = timer.getTime()

    key_left = int(keys[0] in KEYS_LEFT)
    response = int((key_left and is_ll_on_left) or
                   (not key_left and not is_ll_on_left))  # LL option

    draw_option(design['t_ss'], design['r_ss'], -1 * direction, response == 0)
    draw_option(design['t_ll'], design['r_ll'], 1 * direction, response == 1)
    window.flip()
    core.wait(1)

    # Show an empty screen
    window.flip()
    core.wait(1)

    return is_ll_on_left, key_left, response, rt

###############################################################################
# Start PsychoPy
###############################################################################


# Show an information dialog
info = {
    'Number of practices': 5,
    'Number of trials': 20,
}

dialog = gui.DlgFromDict(info, title='Task settings')
if not dialog.OK:
    core.quit()

timestamp = data.getDateStr('%Y%m%d%H%M')
n_trial = int(info['Number of trials'])
n_prac = int(info['Number of practices'])

# Prepare a filename for output data
filename_output = 'ddt_{}.csv'.format(timestamp)
PATH_DATA.mkdir(exist_ok=True)
path_output = PATH_DATA / filename_output

# Open a window
window = visual.Window(size=[1440, 900],
                       units='deg',
                       monitor='testMonitor',
                       color='#333',
                       screen=0,
                       allowGUI=False,
                       fullscr=True)

# Assign the escape key for a shutdown
event.globalKeys.add(key='escape', func=core.quit, name='shutdown')

###############################################################################
# Initialization for ADO
###############################################################################

# Create Task and Model objects for the CRA task
task = TaskDD()
model = ModelHyp()

# Generate grid for design variables and model parameters
grid_design = {
    't_ss': [0],
    't_ll': [0.43, 0.714, 1, 2, 3, 4.3, 6.44, 8.6, 10.8, 12.9,
             17.2, 21.5, 26, 52, 104, 156, 260, 520],
    'r_ss': np.arange(12.5, 800, 12.5),  # [12.5, 25, ..., 787.5]
    'r_ll': [800]
}

grid_param = {
    'k': np.logspace(-5, 0, 50),
    'tau': np.linspace(0, 5, 50)
}

# Initialize an engine
engine = Engine(task, model, grid_design, grid_param)

###############################################################################
# Main codes
###############################################################################

# Make an empty DataFrame to store trial-by-trial information
df_data = pd.DataFrame(
    None,
    columns=['block', 'trial',
             't_ss', 't_ll', 'r_ss', 'r_ll',
             'is_ll_on_left', 'key_left', 'response', 'rt',
             'mean_k', 'mean_tau'])

# Show instructions
show_instruction(INSTRUCTION[0])
show_instruction(INSTRUCTION[1])
show_instruction(INSTRUCTION[2])

# Show countdowns before practices
show_countdown()

# Run practices
for trial in range(n_prac):
    # Get a randomly chosen design
    design = engine.get_design('random')

    # Run a trial using the design
    is_ll_on_left, key_left, response, rt = run_trial(design)

    # Append the current trial into the DataFrame
    df_data = df_data.append(pd.Series({
        'block': 'prac',
        'trial': trial + 1,
        't_ss': design['t_ss'],
        't_ll': design['t_ll'],
        'r_ss': design['r_ss'],
        'r_ll': design['r_ll'],
        'is_ll_on_left': is_ll_on_left,
        'key_left': key_left,
        'response': response,
        'rt': rt,
    }), ignore_index=True)

    # Save the data into a file
    df_data.to_csv(path_output, index=False)

# Show instructions
show_instruction(INSTRUCTION[3])

# Show countdowns before the main task
show_countdown()

# Run the main task
for trial in range(n_trial):
    # Get a design from the ADOpy Engine
    design = engine.get_design()

    # Run a trial using the design
    is_ll_on_left, key_left, response, rt = run_trial(design)

    # Update the engine
    engine.update(design, response)

    # Append the current trial into the DataFrame
    df_data = df_data.append(pd.Series({
        'block': 'main',
        'trial': trial + 1,
        't_ss': design['t_ss'],
        't_ll': design['t_ll'],
        'r_ss': design['r_ss'],
        'r_ll': design['r_ll'],
        'is_ll_on_left': is_ll_on_left,
        'key_left': key_left,
        'response': response,
        'rt': rt,
        'mean_k': engine.post_mean[0],
        'mean_tau': engine.post_mean[1],
        'sd_k': engine.post_sd[0],
        'sd_tau': engine.post_sd[1],
    }), ignore_index=True)

    # Save the data into a file
    df_data.to_csv(path_output, index=False)

# Show the final instruction
show_instruction(INSTRUCTION[4])

window.close()
