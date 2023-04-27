import numpy as np
import matplotlib.pyplot as plt  
x = [0,10,20,30,40,50,60,70,80,90]


# --------------------------------
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ValueError: ',' is not a valid value for ls; supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

# plt.plot(x, baseline, color = '#800080', linestyle='dashed', label="Baseline")
plt.plot(x, trained_with_sparsity, color = 'g', label="Baseline",linestyle='-',lw=2)

plt.plot(x, pruned, color = 'r',label="Pruned",linestyle='--',lw=2, marker='o',markersize=4)
plt.plot(x, fine_tuned, color = 'b',label="Fine-tuned",linestyle='--',lw=2, marker='o',markersize=4)
plt.plot(x, scratch, color = 'm', label="Scratch", linestyle='--',lw=2, marker='o',markersize=4)


plt.xlim([0, 90])
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
plt.axvline(50, color='grey', alpha=0.25)  # median position
plt.tick_params(top=True,bottom=True,left=True,right=True)


plt.xlabel("Pruned Channels (%)")
plt.ylabel("Accuracy (%)")
plt.legend(loc = "best")
plt.savefig('./line_chart_deap.jpg',bbox_inches='tight', pad_inches=0)
plt.savefig('./line_chart_deap.pdf',bbox_inches='tight', pad_inches=0)