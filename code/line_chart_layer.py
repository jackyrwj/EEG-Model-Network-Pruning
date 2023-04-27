import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker
# x = [0,10,20,30,40,50,60,70,80,90]
x = [10,20,30,40,50,60,70,80,90]

# # --------------------------------
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'

# # ValueError: ',' is not a valid value for ls; supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

# # plt.plot(x, baseline, color = '#800080', linestyle='dashed', label="Baseline")
# plt.plot(x, layer1, color = 'g', label="conv_1 64",linestyle='--',marker='o',markersize=5)
# plt.plot(x, layer2, color = 'r',label="conv_2 128",linestyle='--', marker='o',markersize=5)
# plt.plot(x, layer3, color = 'b',label="conv_3 256",linestyle='-', marker='o',markersize=5)
# plt.plot(x, layer4, color = 'm', label="conv_4 64", linestyle='-',marker='o',markersize=5)


# plt.xlim([10,90])
# plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
# plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
# plt.axvline(50, color='grey', alpha=0.25)  # median position
# plt.tick_params(top=True,bottom=True,left=True,right=True)


# plt.xlabel("Pruned Channels (%)")
# plt.ylabel("Accuracy Loss (%)")
# plt.legend(loc = "best")
# plt.savefig('./line_chart_layer_CCNN.jpg',bbox_inches='tight', pad_inches=0)
# plt.savefig('./line_chart_layer_CCNN.pdf',bbox_inches='tight', pad_inches=0)





# --------------------------------
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


plt.plot(x, layer0, color = 'b', label="conv_1 12",linestyle='--',marker='o',markersize=5)
plt.plot(x, layer1, color = 'g', label="conv_2 32",linestyle='--',marker='o',markersize=5)
plt.plot(x, layer2, color = 'r',label="conv_3 64",linestyle='--', marker='o',markersize=5)
plt.plot(x, layer3, color = 'c',label="conv_4 128",linestyle='-', marker='o',markersize=5)
plt.plot(x, layer4, color = 'm', label="conv_5 256", linestyle='-', marker='o',markersize=5)
plt.plot(x, layer5, color = 'y', label="conv_6 128", linestyle='-', marker='o',markersize=5)
plt.plot(x, layer6, color = 'k', label="conv_7 32", linestyle='-', marker='o',markersize=5)


plt.xlim([10, 90])
plt.xticks([ 10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
plt.axvline(50, color='grey', alpha=0.25)  # median position
plt.tick_params(top=True,bottom=True,left=True,right=True)





plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.f'))


plt.xlabel("Pruned Channels (%)")
plt.ylabel("Accuracy Loss (%)")
plt.legend(loc = "best")
plt.savefig('./line_chart_layer_FBCCNN.jpg',bbox_inches='tight', pad_inches=0)
plt.savefig('./line_chart_layer_FBCCNN.pdf',bbox_inches='tight', pad_inches=0)








# # --------------------------------
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'



# # plt.plot(x, temprol_layer, color = 'b', label="temprol_layer 64",linestyle='--',lw=2,marker='o',markersize=4)
# plt.plot(x, temprol_layer, color = 'b', label="temprol_layer 45",linestyle='--',marker='o',markersize=5)
# plt.plot(x, spatial_layer, color = 'g', label="spatial_layer 30",linestyle='--',marker='o',markersize=5)
# plt.plot(x, fusion_layer, color = 'r',label="fusion_layer 15",linestyle='-', marker='o',markersize=5)


# plt.xlim([10, 90])
# plt.xticks([ 10, 20, 30, 40, 50, 60, 70, 80, 90])
# plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
# plt.axvline(50, color='grey', alpha=0.25)  # median position
# plt.tick_params(top=True,bottom=True,left=True,right=True)


# plt.xlabel("Pruned Channels (%)")
# plt.ylabel("Accuracy Loss (%)")
# plt.legend(loc = "best")
# plt.savefig('./line_chart_layer_tsception.jpg',bbox_inches='tight', pad_inches=0)
# plt.savefig('./line_chart_layer_tsception.pdf',bbox_inches='tight', pad_inches=0)



