import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
from ABseq_func import *
import config
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

subject = config.subjects_list[0]
epo_items = epoching_funcs.load_epochs_items(subject)

# ------ now keep only the metadata ----
meta = epo_items.metadata

# the goal is to determine if there is a correlation between complexity and the structure for each of the 16 positions
# ChunkBeginning
# ChunkEnd
# ChunkDepth
# OpenedChunks

fields = ['Complexity','SequenceID','ChunkBeginning','ChunkEnd','ChunkDepth','OpenedChunks']

meta_struct = meta[fields]

# sns.pairplot(meta_struct,hue='Complexity')

results = {'ChunkBeginning':[], 'ChunkEnd':[], 'ChunkDepth':[], 'OpenedChunks':[]}

for fields in ['ChunkBeginning', 'ChunkEnd', 'ChunkDepth', 'OpenedChunks']:
    for stimpos in range(1,17):
        meta_pos = meta.query("StimPosition == %i"%stimpos)
        struct_vals = meta_pos[fields].values
        comp_vals = meta_pos["Complexity"].values
        r = np.corrcoef(struct_vals,comp_vals)
        print("The pearson correlation between %s and the complexity is"%fields)
        print(r)
        results[fields].append(r[0,1])

plt.figure(1)
plt.subplot(411)
plt.plot(results['ChunkBeginning'])
plt.subplot(412)
plt.plot(results['ChunkEnd'])
plt.subplot(413)
plt.plot(results['ChunkDepth'])
plt.subplot(414)
plt.plot(results['OpenedChunks'])
plt.show()







