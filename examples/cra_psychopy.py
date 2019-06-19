import random
from pathlib import Path

import numpy as np
from scipy.stats import bernoulli
import pandas as pd
from psychopy import core, visual, event, data, gui

from adopy import Engine
from adopy.tasks.cra import TaskCRA, ModelLinear

###############################################################################
# Global variables
###############################################################################

PATH_DATA = Path('./data')

BOX_W = 3
BOX_H = 9
DIST_BTWN = 12

LINE_WIDTH = 8
LINE_COLOR = '#ffffff'
COLOR_RED = '#ff0033'
COLOR_BLUE = '#0010ff'
COLOR_GRAY = '#939393'
COLOR_HIDDEN = '#555555'

TEXT_FONT = 'Arial'
TEXT_SIZE = 10
TEXT_MARGIN = 1

FIXATION_SIZE = 1

KEYS_LEFT = ['left', 'z', 'f']
KEYS_RIGHT = ['right', 'slash', 'j']

INSTRUCTION = [
    """
This task is the choice under risk and ambiguity task.
On every trial, two different options will be presented on the screen.
For each option, it has the reward you can earn on the upper side.


Press <space> to proceed.
""", """
You can win the reward with a probability in proportion to how much
the red area occupies in the entire box.

In some cases, the probability can be hidden by a gray box.
Then, the probability will be chosen randomly within the hidden area.

Press <space> to proceed.
""", """
You can choose one by pressing <f> (left option) or <j> (right option).
It will not be presented how much you earn by your choice for each trial.
After finishing the task, the total reward you earn will be presented.
Please make your best to maximize the total reward.


Press <space> to proceed.
""", """
Let's do some practices to check if you understand the task.

Press <space> to start practices.
""", """
Great job. Now, Let's get into the main task.

Press <space> to start a main game.
""", """
Great job. The total reward you have earned is {}.

Press <space> to end.
""",
]


###############################################################################
# Functions for the CRA task
###############################################################################


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


def draw_option(window, lr, is_top, prob, ambig, reward):
    assert lr in [1, -1]
    assert 0 <= prob <= 1
    assert 0 <= ambig <= 1

    x_center = lr * DIST_BTWN / 2
    x_left = x_center - BOX_W
    x_right = x_center + BOX_W
    y_top = BOX_H / 2
    y_bottom = -y_top

    if is_top:
        y_prob = y_top - BOX_H * prob
    else:
        y_prob = y_bottom + BOX_H * prob

    # Draw the blue box
    box_blue = visual.ShapeStim(window, fillColor=COLOR_BLUE,
                                lineWidth=LINE_WIDTH, lineColor=LINE_COLOR,
                                vertices=((x_left, y_top),
                                          (x_right, y_top),
                                          (x_right, y_bottom),
                                          (x_left, y_bottom)))
    box_blue.draw()

    # Draw the red box
    box_red = visual.ShapeStim(window, fillColor=COLOR_RED,
                               lineWidth=LINE_WIDTH, lineColor=LINE_COLOR,
                               vertices=((x_left, y_prob),
                                         (x_right, y_prob),
                                         (x_right, y_bottom),
                                         (x_left, y_bottom)))
    box_red.draw()

    # Draw the gray box if ambig > 0
    if ambig > 0:
        x_ambig_left = x_center - BOX_W * 1.1
        x_ambig_right = x_center + BOX_W * 1.1
        y_ambig_top = BOX_H / 2 * ambig
        y_ambig_bottom = -y_ambig_top

        box_ambig = visual.ShapeStim(window, fillColor=COLOR_GRAY,
                                     lineWidth=LINE_WIDTH, lineColor=LINE_COLOR,
                                     vertices=((x_ambig_left, y_ambig_top),
                                               (x_ambig_right, y_ambig_top),
                                               (x_ambig_right, y_ambig_bottom),
                                               (x_ambig_left, y_ambig_bottom)))
        box_ambig.draw()

    if is_top:
        r_top, r_bottom = reward, 0
    else:
        r_top, r_bottom = 0, reward

    text_top = visual.TextStim(window, text='{:.0f}'.format(r_top),
                               pos=(x_center, y_top + TEXT_MARGIN))
    text_top.size = TEXT_SIZE
    text_top.draw()

    text_bottom = visual.TextStim(window, text='{:.0f}'.format(r_bottom),
                                  pos=(x_center, y_bottom - TEXT_MARGIN))
    text_bottom.size = TEXT_SIZE
    text_bottom.draw()


def draw_hidden_option(window, lr):
    assert lr in [1, -1]

    x_center = lr * DIST_BTWN / 2
    x_left = x_center - BOX_W * 1.1
    x_right = x_center + BOX_W * 1.1
    y_top = BOX_H / 2 + 2 * TEXT_MARGIN
    y_bottom = -y_top

    box = visual.ShapeStim(window, fillColor=COLOR_HIDDEN,
                           lineWidth=LINE_WIDTH, lineColor=LINE_COLOR,
                           vertices=((x_left, y_top),
                                     (x_right, y_top),
                                     (x_right, y_bottom),
                                     (x_left, y_bottom)))
    box.draw()


def run_trial(design):
    """
    Run a single trial for the CRA task based on the given design.
    """
    global window

    # Show a fixation cross
    fixation = visual.GratingStim(window, color='white', tex=None,
                                  mask='cross', size=FIXATION_SIZE)
    fixation.draw()
    window.flip()
    core.wait(1)

    # Randomize the order of two options
    lr = random.randint(0, 1) * 2 - 1

    # Randomize whether the reward is placed on top
    is_top = random.randint(0, 1)

    # Show a variable option
    draw_option(window, 1 * lr, is_top,
                design['p_var'], design['a_var'], design['r_var'])

    # Show a fixed option (a reference option)
    draw_option(window, -1 * lr, is_top, 0.5, 0, design['r_fix'])

    # Show two options for 2 seconds
    window.flip()
    core.wait(2)

    # Draw hidden options
    draw_hidden_option(window, 1)
    draw_hidden_option(window, -1)
    window.flip()

    # wait a response
    timer = core.Clock()
    keys = event.waitKeys(keyList=KEYS_LEFT + KEYS_RIGHT)
    rt = timer.getTime()

    # Show the chosen option
    lr_chosen = -1 if keys[0] in KEYS_LEFT else 1
    draw_hidden_option(window, lr_chosen)
    window.flip()
    core.wait(1)

    resp = int(lr == lr_chosen)

    return (resp, rt)


def generate_fixed_designs():
    """
    Return design pairs used by Levy et al. (2010)
    """
    # For risky conditions
    pr_risky = np.array([.13, .25, .38])
    am_risky = np.array([.0])

    # For ambiguous conditions
    pr_ambig = np.array([.5])
    am_ambig = np.array([.25, .50, .75])

    # Make cartesian products for each condition
    pr_am_risky = np.squeeze(np.stack(np.meshgrid(pr_risky, am_risky), -1))
    pr_am_ambig = np.squeeze(np.stack(np.meshgrid(pr_ambig, am_ambig), -1))

    # Merge two grids into one object
    pr_am = np.vstack([pr_am_risky, pr_am_ambig])

    rv = np.array([5, 9.5, 18, 34, 65])
    rf = np.array([5])

    rewards = np.vstack([(v, f) for v in rv for f in rf])

    designs = np.array([
        np.concatenate([pr_am[i], rewards[j], [k]])
        for i in range(len(pr_am))
        for j in range(len(rewards))
        for k in range(2)
    ])
    np.random.shuffle(designs)

    return pd.DataFrame(designs[:, :4],
                        columns=['p_var', 'a_var', 'r_var', 'r_fix'])

###############################################################################
# Start PsychoPy
###############################################################################


# Show an information dialog
info = {
    'Number of practices': 5,
    'Number of trials': 60,
}

dialog = gui.DlgFromDict(info, title='Task settings')
if not dialog.OK:
    core.quit()

timestamp = data.getDateStr('%Y%m%d%H%M')
n_trial = int(info['Number of trials'])
n_prac = int(info['Number of practices'])

# Prepare a filename for output data
filename_output = 'cra_{}.csv'.format(timestamp)
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
# Prepare ADOpy objects
###############################################################################

# Create Task and Model objects for the CRA task
task = TaskCRA()
model = ModelLinear()

# p_var & a_var for risky & ambiguous trials
pval = [.05, .10, .15, .20, .25, .30, .35, .40, .45]
aval = [.125, .25, .375, .5, .625, .75]

# risky trials: a_var fixed to 0
pa_risky = [[p, 0] for p in pval]
# ambiguous trials: p_var fixed to 0.5
pa_ambig = [[0.5, a] for a in aval]
pr_am = np.array(pa_risky + pa_ambig)

# r_var & r_fix while r_var > r_fix
rval = [10, 15, 21, 31, 45, 66, 97, 141, 206, 300]
rewards = []
for r_var in rval:
    for r_fix in rval:
        if r_var > r_fix:
            rewards.append([r_var, r_fix])
rewards = np.array(rewards)

grid_design = {('p_var', 'a_var'): pr_am, ('r_var', 'r_fix'): rewards}

grid_param = {
    'alpha': np.linspace(0, 3, 11),
    'beta': np.linspace(-3, 3, 11),
    'gamma': np.linspace(0, 5, 11)
}

# Create an Engine object
engine = Engine(task, model, grid_design, grid_param)

###############################################################################
# Main codes
###############################################################################

# Generate pre-determined designs (Levy et al., 2010)
df_fixed = generate_fixed_designs()

# Make an empty DataFrame to store trial-by-trial information
df_data = pd.DataFrame(
    None,
    columns=['block', 'trial', 'p_var', 'a_var', 'r_var', 'r_fix',
             'response', 'rt'])

# Show instructions
show_instruction(INSTRUCTION[0])
show_instruction(INSTRUCTION[1])
show_instruction(INSTRUCTION[2])
show_instruction(INSTRUCTION[3])

# Show countdowns before practices
show_countdown()

# Run practices
for trial in range(n_prac):
    # Get a design from the fixed designs
    design = df_fixed.iloc[trial % len(df_fixed), :]

    # Run a trial using the design
    response, rt = run_trial(design)

    # Append the current trial into the DataFrame
    df_data = df_data.append(pd.Series({
        'block': 'prac',
        'trial': trial + 1,
        'p_var': design['p_var'],
        'a_var': design['a_var'],
        'r_var': design['r_var'],
        'r_fix': design['r_fix'],
        'response': response,
        'rt': rt,
    }), ignore_index=True)

    # Save the data into a file
    df_data.to_csv(path_output, index=False)

# Re-shuffle the fixed designs
df_fixed = df_fixed.sample(frac=1).reset_index(drop=True)

# Show instructions
show_instruction(INSTRUCTION[4])

# Show countdowns before the main task
show_countdown()

# Run the main task
for trial in range(n_trial):
    # Get a design from the ADOpy Engine
    design = engine.get_design()

    # Run a trial using the design
    response, rt = run_trial(design)

    # Update the engine
    engine.update(design, response)

    # Append the current trial into the DataFrame
    df_data = df_data.append(pd.Series({
        'block': 'main',
        'trial': trial + 1,
        'p_var': design['p_var'],
        'a_var': design['a_var'],
        'r_var': design['r_var'],
        'r_fix': design['r_fix'],
        'response': response,
        'rt': rt,
        'mean_alpha': engine.post_mean[0],
        'mean_beta': engine.post_mean[1],
        'mean_gamma': engine.post_mean[2],
        'sd_alpha': engine.post_sd[0],
        'sd_beta': engine.post_sd[1],
        'sd_gamma': engine.post_sd[2],
    }), ignore_index=True)

    # Save the data into a file
    df_data.to_csv(path_output, index=False)

# Compute the total reward
total_reward = 0.
df_main = df_data.loc[df_data['block'] == 'main', :]

df_main_fix = df_main.loc[df_main['response'] == 0, :]
num_resp_fix = len(df_main_fix)

if num_resp_fix > 0:
    earn = bernoulli.rvs(np.repeat(0.5, num_resp_fix))
    total_reward += np.dot(earn, df_main_fix['r_fix'])

df_main_var = df_main.loc[df_main['response'] == 1, :]
num_resp_var = len(df_main_var)

if num_resp_var > 0:
    earn = bernoulli.rvs(df_main_var['p_var'])
    total_reward += np.dot(earn, df_main_var['r_var'])

# Show the final instruction
show_instruction(INSTRUCTION[5].format(int(total_reward)))

window.close()
