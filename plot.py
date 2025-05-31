import torch
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import matplotlib.gridspec as gridspec
from datasets import load_dataset
import numpy as np
from transformers import pipeline
from sklearn.manifold import MDS
from matplotlib.lines import Line2D
import pandas as pd

gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])

fig = plt.figure(figsize=(0.78*7.20472, 2))
plt.style.use(['nature'])

ax1 = fig.add_subplot(gs[:, 0])

num_models = 3
num_runs = 10
num_iterations = 5
num_participants = 6

AICs = np.zeros((num_models, num_runs, num_iterations, num_participants))

for model_id in range(num_models):
    for simulation in range(num_runs):
        for i in range(5):
            data = np.load('data/srm_model_' + str(model_id) + '_run_' + str(simulation) + '_iteration_' + str(i) + '.npz')
            AICs[model_id, simulation, i, :] = 2 * data['num_parameters'].item() + 2 * data['nll'].sum(-1)

AICs_sum_over_participts = AICs.sum(-1)

best_n = 10
flat = AICs_sum_over_participts.ravel()
sorted_idx = np.argsort(flat)[:best_n]
multi_idx = np.unravel_index(sorted_idx, AICs_sum_over_participts.shape)

for val, idx in zip(flat[sorted_idx], zip(*multi_idx)):
    print(idx)
    print(val)
    data = np.load('data/srm_model_' + str(idx[0]) + '_run_' + str(idx[1]) + '_iteration_' + str(idx[2]) + '.npz')
    print(data['model_string'])
    #print(AICs_sum_over_participts[idx])

for i in range(AICs_sum_over_participts.shape[0]):
    for j in range(AICs_sum_over_participts.shape[1]):
        best_AIC = AICs_sum_over_participts[i, j, 0]
        for k in range(1, AICs_sum_over_participts.shape[2]):
            if best_AIC < AICs_sum_over_participts[i, j, k]:
                AICs_sum_over_participts[i, j, k] = best_AIC
            else:
                best_AIC = AICs_sum_over_participts[i, j, k]



mean_iter = AICs_sum_over_participts.mean(axis=(0, 1))   # shape (num_iterations,)
std_iter = AICs_sum_over_participts.std(axis=(0, 1))   # shape (num_iterations,)
print(mean_iter)
print(std_iter)
min_iter  = AICs_sum_over_participts.min (axis=(0, 1))   # shape (num_iterations,)
max_iter  = AICs_sum_over_participts.max (axis=(0, 1))   # shape (num_iterations,)
iterations = np.arange(num_iterations)   # [0, 1, 2, 3, 4]
ax1.fill_between(iterations, min_iter, max_iter, alpha=0.15,)

# central tendency line
ax1.plot(iterations, mean_iter, marker='o', lw=1)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("AIC")
ax1.axhline(y=72.5, color='gray', linestyle='--', linewidth=1, label='Centaur')
ax1.set_ylim(0, 400)
ax1.set_xlim(0, 4)
ax1.set_xticks(np.arange(num_iterations))

ax1.text(
    x=0.4,
    y=72.5 + 0.5,
    s='Centaur',
    ha='center',
    va='bottom',
    fontsize=7,
    color='gray'
)

ax2 = fig.add_subplot(gs[:, 1])

data_first = [AICs[:, :, 0, p].ravel() for p in range(num_participants)]   # iteration 0
data_last  = [AICs[:, :, 4, p].ravel() for p in range(num_participants)]   # iteration 4

positions_first = np.arange(num_participants) - 0.2
positions_last  = np.arange(num_participants) + 0.2

# draw the two boxplot groups
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'white'},
    'medianprops':{'color':'white'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

bp1 = ax2.boxplot(data_first, positions=positions_first, widths=0.35, patch_artist=True, showfliers=False, **PROPS)
bp2 = ax2.boxplot(data_last,  positions=positions_last,  widths=0.35, patch_artist=True, showfliers=False, **PROPS)

# colour them to match your old bar colours
for box in bp1['boxes']:
    box.set(facecolor='C0',)
for box in bp2['boxes']:
    box.set(facecolor='C1')

for median in bp1['medians']:
    median.set(color='white')
for median in bp2['medians']:
    median.set(color='white')
# aesthetics
ax2.set_xticks(np.arange(num_participants), np.arange(num_participants))
ax2.set_xlabel('Participant')
ax2.set_ylabel('AIC')

custom_lines = [
    Line2D([0], [0], color='C0', linewidth=5, label='First iteration'),
    Line2D([0], [0], color='C1', linewidth=5, label='Last iteration')
]
ax2.legend(handles=custom_lines, frameon=False)


sns.despine()
plt.tight_layout()
plt.savefig('figures/fig2.pdf', bbox_inches='tight')
plt.show()
