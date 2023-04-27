
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset, SEEDDataset
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LIST, DEAP_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import (KFoldCrossTrial, KFoldGroupbyTrial,
                                      KFoldPerSubject, train_test_split)
from torcheeg.model_selection.k_fold import KFold
from compute_flops import count_model_param_flops, print_model_param_nums
from torcheeg1.models import TSCeption
from torcheeg1.models import CCNN
from torcheeg import transforms
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np


labels = ['LFP', 'UWP', 'SFP']
Origin = [100,  100, 100]
Parameter = [83.414, 14.052, 11.838]
FLOPs  = [95, 88.75, 74.375]

x = np.arange(len(labels))  # the label locations
width = 0.28  # the width of the bars

fig, ax = plt.subplots(figsize=(6.4,4))
rects1 = ax.bar(x - width, Origin, width, label='Origin',color=['darkorange'])
rects2 = ax.bar(x , Parameter, width, label='Parameter',color=['blue'])
rects3 = ax.bar(x +  width, FLOPs, width, label='FLOPs',color=['green'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Ratio(%)')
ax.set_title('CCNN')
ax.set_xticks(x, labels)
ax.legend(loc='upper center', bbox_to_anchor=(1,1.05),fancybox=True,shadow=True)

ax.bar_label(rects1, labels=['%.1f%%' % e for e in Origin],padding=1,fontsize=8)
ax.bar_label(rects2, labels=['%.1f%%' % e for e in Parameter],padding=1,fontsize=8)
ax.bar_label(rects3, labels=['%.1f%%' % e for e in FLOPs],padding=1,fontsize=8)

fig.tight_layout()

plt.savefig('./compare_deap_CCNN.jpg')
plt.savefig('./compare_deap_CCNN.pdf')








