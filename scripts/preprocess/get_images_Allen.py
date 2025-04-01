#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_stimulus_table(stim_table, title):
    fstart = stim_table.start.min()
    fend = stim_table.end.max()
    
    fig = plt.figure(figsize=(15,1))
    ax = fig.gca()
    for i, trial in stim_table.iterrows():    
        x1 = float(trial.start - fstart) / (fend - fstart)
        x2 = float(trial.end - fstart) / (fend - fstart)            
        ax.add_patch(patches.Rectangle((x1, 0.0), x2 - x1, 1.0, color='r'))
    ax.set_xticks((0,1))
    ax.set_xticklabels((fstart, fend))
    ax.set_yticks(())
    ax.set_title(title)
    ax.set_xlabel("frames")

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
data_set = boc.get_ophys_experiment_data(501940850)

# this is a pandas DataFrame. find trials with a given stimulus condition.
temporal_frequency = 4
orientation = 225
stim_table = data_set.get_stimulus_table('drifting_gratings')
stim_table = stim_table[(stim_table.temporal_frequency == temporal_frequency) & (stim_table.orientation == orientation)]

# plot the trials
plot_stimulus_table(stim_table, "TF %d ORI %d" % (temporal_frequency, orientation))

data_set = boc.get_ophys_experiment_data(501498760)

scene_nums = [4, 83]

# read in the array of images
scenes = data_set.get_stimulus_template('natural_scenes')

# display a couple of the scenes
fig, axes = plt.subplots(1,len(scene_nums))
for ax,scene in zip(axes, scene_nums):
    ax.imshow(scenes[scene,:,:], cmap='gray')
    ax.set_axis_off()
    ax.set_title('scene %d' % scene)