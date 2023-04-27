
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

# plt.rcdefaults()
# fig, ax = plt.subplots()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.09, hspace=0.3)
plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.08, hspace=None)
# ----------------------------
# ax1.spines['right'].set_color('none')
# ax1.spines['top'].set_color('none') 

# ax2.spines['left'].set_color('none')
# ax2.spines['top'].set_color('none') 
# ----------------------------
# Example data
methods = ('LFP','UMWP','SFP')
y_pos = np.arange(len(methods))
# Parameters = [34.25, 11.84,11.79]
# FLOPs = [50.0,74.38,74.38]
# Parameters = [34.25, 11.84,10.24]
# FLOPs = [50.0,74.38,69.52]
Parameters = [34.25, 11.84,11.29]
FLOPs = [50.00,28.13,27.50]

#------------------------------
rects = ax1.barh(y_pos, Parameters, align='center',color='red', ecolor='black',height=0.5)
ax1.set_yticks(y_pos, labels=methods)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.tick_params( axis = 'x',direction  = 'in', labelsize = 15)
ax1.tick_params( axis = 'y',direction  = 'in', labelsize = 20)


ax1.bar_label(rects,  labels=[f'{x:,.1f}%' for x in Parameters], padding=3, color='black', fontweight='bold', fontsize = 20)
ax1.set_xlim([0, 100])
ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax1.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax1.axvline(50, color='grey', alpha=0.25)  # median position
ax1.set_title('Parameters Saving' , fontdict = {'fontsize': 20})
ax1.set_xlabel('Ratio(%)', fontsize = '20')

#------------------------------
ax7 = ax2.twinx() # Create a twin x-axis
ax7.tick_params( axis = 'y', direction  = 'in', labelsize = 20)

# important
# gca = plt.gca()
# gca.spines['left'].set_color('none')
# gca.spines['top'].set_color('none') 

rects = ax7.barh(y_pos, FLOPs,  align='center',color='blue', ecolor='black',height=0.5) # Plot using `ax1` instead of `ax`
ax7.set_yticks(y_pos,labels=methods)
ax7.invert_yaxis()  # labels read top-to-bottom
ax7.set_title('FLOPs Saving', fontdict = {'fontsize': 20})

ax7.bar_label(rects,  labels=[f'{x:,.1f}%' for x in FLOPs], padding=-70, color='black', fontweight='bold', fontsize = 20)
ax7.set_xlim([0, 100])
ax7.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax2.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax7.axvline(50, color='grey', alpha=0.25)  # median position

ax2.invert_xaxis()
ax2.set_yticklabels([]) # Hide the left y-axis tick-labels
ax2.set_yticks([]) # Hide the left y-axis ticks

ax2.tick_params( axis='x', direction  = 'in' ,labelsize = 15)
ax2.set_xlabel('Ratio(%)', fontsize = '20')
# -----------------------------------------------
plt.savefig('./CCNN.jpg', bbox_inches='tight', pad_inches=0)
plt.savefig('./CCNN.pdf', bbox_inches='tight', pad_inches=0)








fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.08, hspace=None)
# ----------------------------
# ax3.spines['right'].set_color('none')
# ax3.spines['top'].set_color('none') 

# ax4.spines['left'].set_color('none')
# ax4.spines['top'].set_color('none') 

# ----------------------------
# Example data
methods = ('LFP','NSFP','UMWP','SFP')
y_pos = np.arange(len(methods))
# Parameters = [37.21,37.28,27.29,27.15]
# FLOPs = [50.83,49.17,78.33,77.50]
# Parameters = [14.50,5.34,3.40,3.26]
# FLOPs = [19.17,12.00,10.00,9.17]
Parameters = [37.21,37.18,10.25,10.10]
FLOPs = [50.83,49.17,29.17,29.17]

#------------------------------
rects = ax3.barh(y_pos, Parameters, align='center',color='red', ecolor='black',height=0.5)
ax3.set_yticks(y_pos, labels=methods)
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.tick_params( axis = 'x', direction  = 'in', labelsize = 15)
ax3.tick_params( axis = 'y', direction  = 'in', labelsize = 20)

ax3.bar_label(rects, labels=[f'{x:,.1f}%' for x in Parameters], padding=3, color='black', fontweight='bold', fontsize = 20)
ax3.set_xlim([0, 100])
ax3.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax3.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax3.axvline(50, color='grey', alpha=0.25)  # median position
ax3.set_title('Parameters Saving' , fontdict = {'fontsize': 20})
ax3.set_xlabel('Ratio(%)', fontsize = '20')

#------------------------------
ax7 = ax4.twinx() # Create a twin x-axis
ax7.tick_params( axis = 'y', direction  = 'in', labelsize = 20)

# gca = plt.gca()
# gca.spines['left'].set_color('none')
# gca.spines['top'].set_color('none') 

rects = ax7.barh(y_pos, FLOPs,  align='center',color='blue', ecolor='black',height=0.5) # Plot using `ax1` instead of `ax`
ax7.set_yticks(y_pos,labels=methods)
ax7.invert_yaxis()  # labels read top-to-bottom
ax7.set_title('FLOPs Saving',  fontdict = {'fontsize': 20})

ax7.bar_label(rects, labels=[f'{x:,.1f}%' for x in FLOPs], padding=-70, color='black', fontweight='bold', fontsize = 20)
ax7.set_xlim([0, 100])
ax7.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax4.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
ax7.axvline(50, color='grey', alpha=0.25)  # median position


ax4.invert_xaxis()
ax4.set_yticklabels([]) # Hide the left y-axis tick-labels
ax4.set_yticks([]) # Hide the left y-axis ticks

ax4.tick_params( axis='x', direction  = 'in' ,labelsize = 15)
ax4.set_xlabel('Ratio(%)', fontsize = '20')
# -----------------------------------------------
plt.savefig('./FBCCNN.jpg', bbox_inches='tight', pad_inches=0)
plt.savefig('./FBCCNN.pdf', bbox_inches='tight', pad_inches=0)




















