

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

# ----------------------------
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none') 

ax2.spines['left'].set_color('none')
ax2.spines['top'].set_color('none') 

# ----------------------------
# Example data
methods = ('Baseline','LFP','NSFP','UWP','SFP')
y_pos = np.arange(len(methods))
fine_tuned = [92.820, 93.258, 93.490, 93.924, 92.618]
scratch = [92.820, 93.056, 92.802, 91.412, 91.266]

#------------------------------
rects = ax1.barh(y_pos, fine_tuned, align='center',color='red', ecolor='black',height=0.3)
ax1.set_yticks(y_pos, labels=methods)
ax1.invert_yaxis()  # labels read top-to-bottom


ax1.bar_label(rects, fine_tuned,padding=3, color='black', fontweight='bold')
ax1.set_xlim([0, 100])
ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax1.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax1.axvline(50, color='grey', alpha=0.25)  # median position
ax1.set_title('Find-tuned Accuracy(%)')

#------------------------------
ax7 = ax2.twinx() # Create a twin x-axis
gca = plt.gca()
gca.spines['left'].set_color('none')
gca.spines['top'].set_color('none') 

rects = ax7.barh(y_pos, scratch,  align='center',color='blue', ecolor='black',height=0.3) # Plot using `ax1` instead of `ax`
ax7.set_yticks(y_pos,labels=methods)
ax7.invert_yaxis()  # labels read top-to-bottom
# ax7.set_title('FLOPs Savings(%)')
ax7.set_title('Scratch Accuracy(%)')

ax7.bar_label(rects, scratch,padding=-40, color='black', fontweight='bold')
ax7.set_xlim([0, 100])
ax7.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax2.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax7.axvline(50, color='grey', alpha=0.25)  # median position


ax2.invert_xaxis()
ax2.set_yticklabels([]) # Hide the left y-axis tick-labels
ax2.set_yticks([]) # Hide the left y-axis ticks

# -----------------------------------------------
# plt.savefig('./test.jpg')
plt.suptitle('FBCCNN_10%')
plt.rcParams['xtick.direction'] = 'in'   # 刻度线内向
plt.rcParams['ytick.direction'] = 'in'
# plt.savefig('./CCNN.jpg', bbox_inches='tight', pad_inches=0)
# plt.savefig('./CCNN.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('accracy.jpg')