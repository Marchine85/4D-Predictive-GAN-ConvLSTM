import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT CONVLSTM MODEL ERRORS
errors_t20_inc1 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T20_TINC1.txt"), delimiter=",", skiprows=1)
errors_t40_inc2 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T40_TINC2.txt"), delimiter=",", skiprows=1)
errors_t80_inc4 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T80_TINC4.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT MSE ERRORS
###############################################################################
mse_t20_inc1 = errors_t20_inc1[:,0]
mse_t20_inc1_mean = np.round(np.mean(mse_t20_inc1),4)
mse_t20_inc1_stddev = np.round(np.std(mse_t20_inc1),4)
mse_null_t20_inc1 = errors_t20_inc1[:,1]
mse_null_t20_inc1_mean = np.round(np.mean(mse_null_t20_inc1),4)
mse_null_t20_inc1_stddev = np.round(np.std(mse_null_t20_inc1),4)
mse_input_t20_inc1 = errors_t20_inc1[:,2]
mse_input_t20_inc1_mean = np.round(np.mean(mse_input_t20_inc1),4)
mse_input_t20_inc1_stddev = np.round(np.std(mse_input_t20_inc1),4)

mse_t40_inc2 = errors_t40_inc2[:,0]
mse_t40_inc2_mean = np.round(np.mean(mse_t40_inc2),4)
mse_t40_inc2_stddev = np.round(np.std(mse_t40_inc2),4)
mse_null_t40_inc2 = errors_t40_inc2[:,1]
mse_null_t40_inc2_mean = np.round(np.mean(mse_null_t40_inc2),4)
mse_null_t40_inc2_stddev = np.round(np.std(mse_null_t40_inc2),4)
mse_input_t40_inc2 = errors_t40_inc2[:,2]
mse_input_t40_inc2_mean = np.round(np.mean(mse_input_t40_inc2),4)
mse_input_t40_inc2_stddev = np.round(np.std(mse_input_t40_inc2),4)

mse_t80_inc4 = errors_t80_inc4[:,0]
mse_t80_inc4_mean = np.round(np.mean(mse_t80_inc4),4)
mse_t80_inc4_stddev = np.round(np.std(mse_t80_inc4),4)
mse_null_t80_inc4 = errors_t80_inc4[:,1]
mse_null_t80_inc4_mean = np.round(np.mean(mse_null_t80_inc4),4)
mse_null_t80_inc4_stddev = np.round(np.std(mse_null_t80_inc4),4)
mse_input_t80_inc4 = errors_t80_inc4[:,2]
mse_input_t80_inc4_mean = np.round(np.mean(mse_input_t80_inc4),4)
mse_input_t80_inc4_stddev = np.round(np.std(mse_input_t80_inc4),4)

fig, axs = plt.subplots(3, 4, sharex=False, sharey=False, layout="constrained", figsize=(12, 8))
colors = ["crimson", "navy", "seagreen"]


labels = [f"µ={mse_t20_inc1_mean}, σ={mse_t20_inc1_stddev}", 
          f"µ={mse_input_t20_inc1_mean}, σ={mse_input_t20_inc1_stddev}",
          f"µ={mse_null_t20_inc1_mean}, σ={mse_null_t20_inc1_stddev}"]
data = [mse_t20_inc1, mse_input_t20_inc1, mse_null_t20_inc1]
axs[0,0].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[0,0].set_title("MSE ERRORS\nT=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].legend(loc="upper right") 
axs[0,0].set_ylabel('Count')
axs[0,0].set_xlim([0,0.05])
#axs[0,0].set_ylim([0,200])

labels = [f"µ={mse_t40_inc2_mean}, σ={mse_t40_inc2_stddev} ",  
          f"µ={mse_input_t40_inc2_mean}, σ={mse_input_t40_inc2_stddev} ",
          f"µ={mse_null_t40_inc2_mean}, σ={mse_null_t40_inc2_stddev} ",]
data = [mse_t40_inc2, mse_input_t40_inc2, mse_null_t40_inc2]
axs[1,0].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[1,0].set_title("T=40 TINC=2")
axs[1,0].grid(True)
axs[1,0].legend(loc="upper right") 
axs[1,0].set_ylabel('Count')
axs[1,0].set_xlim([0,0.05])
#axs[1,0].set_ylim([0,200])

labels = [f"µ={mse_t80_inc4_mean}, σ={mse_t80_inc4_stddev} ",
          f"µ={mse_input_t80_inc4_mean}, σ={mse_input_t80_inc4_stddev} ",
          f"µ={mse_null_t80_inc4_mean}, σ={mse_null_t80_inc4_stddev} ",]
data = [mse_t80_inc4, mse_input_t80_inc4, mse_null_t80_inc4]
axs[2,0].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[2,0].set_title("T=80 TINC=4")
axs[2,0].grid(True)
axs[2,0].legend(loc="upper right") 
axs[2,0].set_xlabel('Error')
axs[2,0].set_ylabel('Count')
axs[2,0].set_xlim([0,0.05])
#axs[2,0].set_ylim([0,200])

###############################################################################
# PLOT JACCARD ERRORS
###############################################################################
jaccard_t20_inc1 = errors_t20_inc1[:,3]
jaccard_t20_inc1_mean = np.round(np.mean(jaccard_t20_inc1),4)
jaccard_t20_inc1_stddev = np.round(np.std(jaccard_t20_inc1),4)
jaccard_null_t20_inc1 = errors_t20_inc1[:,4]
jaccard_null_t20_inc1_mean = np.round(np.mean(jaccard_null_t20_inc1),4)
jaccard_null_t20_inc1_stddev = np.round(np.std(jaccard_null_t20_inc1),4)
jaccard_input_t20_inc1 = errors_t20_inc1[:,5]
jaccard_input_t20_inc1_mean = np.round(np.mean(jaccard_input_t20_inc1),4)
jaccard_input_t20_inc1_stddev = np.round(np.std(jaccard_input_t20_inc1),4)

jaccard_t40_inc2 = errors_t40_inc2[:,3]
jaccard_t40_inc2_mean = np.round(np.mean(jaccard_t40_inc2),4)
jaccard_t40_inc2_stddev = np.round(np.std(jaccard_t40_inc2),4)
jaccard_null_t40_inc2 = errors_t40_inc2[:,4]
jaccard_null_t40_inc2_mean = np.round(np.mean(jaccard_null_t40_inc2),4)
jaccard_null_t40_inc2_stddev = np.round(np.std(jaccard_null_t40_inc2),4)
jaccard_input_t40_inc2 = errors_t40_inc2[:,5]
jaccard_input_t40_inc2_mean = np.round(np.mean(jaccard_input_t40_inc2),4)
jaccard_input_t40_inc2_stddev = np.round(np.std(jaccard_input_t40_inc2),4)

jaccard_t80_inc4 = errors_t80_inc4[:,3]
jaccard_t80_inc4_mean = np.round(np.mean(jaccard_t80_inc4),4)
jaccard_t80_inc4_stddev = np.round(np.std(jaccard_t80_inc4),4)
jaccard_null_t80_inc4 = errors_t80_inc4[:,4]
jaccard_null_t80_inc4_mean = np.round(np.mean(jaccard_null_t80_inc4),4)
jaccard_null_t80_inc4_stddev = np.round(np.std(jaccard_null_t80_inc4),4)
jaccard_input_t80_inc4 = errors_t80_inc4[:,5]
jaccard_input_t80_inc4_mean = np.round(np.mean(jaccard_input_t80_inc4),4)
jaccard_input_t80_inc4_stddev = np.round(np.std(jaccard_input_t80_inc4),4)

labels = [f"µ={jaccard_t20_inc1_mean}, σ={jaccard_t20_inc1_stddev}", 
          f"µ={jaccard_input_t20_inc1_mean}, σ={jaccard_input_t20_inc1_stddev}",
          f"µ={jaccard_null_t20_inc1_mean}, σ={jaccard_null_t20_inc1_stddev}"]
data = [jaccard_t20_inc1, jaccard_input_t20_inc1, jaccard_null_t20_inc1]
a = axs[0,1].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[0,1].set_title("JACCARD ERRORS\nT=20 TINC=1")
axs[0,1].grid(True)
axs[0,1].legend(loc="upper right") 
#axs[0,1].set_ylabel('Count')
axs[0,1].set_xlim([0,0.5])
#axs[0,1].set_ylim([0,500])

labels = [f"µ={jaccard_t40_inc2_mean}, σ={jaccard_t40_inc2_stddev} ",  
          f"µ={jaccard_input_t40_inc2_mean}, σ={jaccard_input_t40_inc2_stddev} ",
          f"µ={jaccard_null_t40_inc2_mean}, σ={jaccard_null_t40_inc2_stddev} ",]
data = [jaccard_t40_inc2, jaccard_input_t40_inc2, jaccard_null_t40_inc2]
axs[1,1].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[1,1].set_title("T=40 TINC=2")
axs[1,1].grid(True)
axs[1,1].legend(loc="upper right") 
#axs[1,1].set_ylabel('Count')
axs[1,1].set_xlim([0,0.5])
#axs[1,1].set_ylim([0,500])

labels = [f"µ={jaccard_t80_inc4_mean}, σ={jaccard_t80_inc4_stddev} ",
          f"µ={jaccard_input_t80_inc4_mean}, σ={jaccard_input_t80_inc4_stddev} ",
          f"µ={jaccard_null_t80_inc4_mean}, σ={jaccard_null_t80_inc4_stddev} ",]
data = [jaccard_t80_inc4, jaccard_input_t80_inc4, jaccard_null_t80_inc4]
axs[2,1].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[2,1].set_title("T=80 TINC=4")
axs[2,1].grid(True)
axs[2,1].legend(loc="upper right") 
axs[2,1].set_xlabel('Error')
#axs[2,1].set_ylabel('Count')
axs[2,1].set_xlim([0,0.5])
#axs[2,1].set_ylim([0,500])


###############################################################################
# PLOT MSE ERRORS
###############################################################################
mse_t20_inc1 = errors_t20_inc1[:,0]
mse_null_t20_inc1 = errors_t20_inc1[:,1]
mse_input_t20_inc1 = errors_t20_inc1[:,2]

mse_t40_inc2 = errors_t40_inc2[:,0]
mse_null_t40_inc2 = errors_t40_inc2[:,1]
mse_input_t40_inc2 = errors_t40_inc2[:,2]

mse_t80_inc4 = errors_t80_inc4[:,0]
mse_null_t80_inc4 = errors_t80_inc4[:,1]
mse_input_t80_inc4 = errors_t80_inc4[:,2]

#fig, axs = plt.subplots(3, 2, sharex=True, sharey=False, layout="constrained", figsize=(10, 10))
colors = ["crimson", "navy", "seagreen"]
meanlineprops = dict(linestyle='', linewidth=0.8, color='black', marker="*", markeredgecolor='black', markerfacecolor='black')
medianlineprops = dict(linestyle='-', linewidth=0.8, color='black')

labels = ["Prediction vs Target(xt1)", "Prediction vs Input(xt0)", "Null Model (xt1 vs xt0)"]

data = [mse_t20_inc1, mse_input_t20_inc1, mse_null_t20_inc1]
bplot1 = axs[0,2].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,2].set_title("MSE ERRORS\nT=20 TINC=1")
axs[0,2].grid(True)
axs[0,2].set_ylabel('Error')
axs[0,2].set_xticks([])
#axs[0,0].set_ylim([0,0.05])


data = [mse_t40_inc2, mse_input_t40_inc2, mse_null_t40_inc2]
bplot2 = axs[1,2].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,2].set_title("T=40 TINC=2")
axs[1,2].grid(True)
axs[1,2].set_ylabel('Error')
axs[1,2].set_xticks([])
#axs[1,0].set_ylim([0,0.05])


data = [mse_t80_inc4, mse_input_t80_inc4, mse_null_t80_inc4]
bplot3 = axs[2,2].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[2,2].set_title("T=80 TINC=4")
axs[2,2].grid(True)
axs[2,2].set_ylabel('Error')
axs[2,2].set_xticks([])
#axs[2,0].set_ylim([0,0.05])

# ###############################################################################
# # PLOT JACCARD ERRORS
# ###############################################################################
jaccard_t20_inc1 = errors_t20_inc1[:,3]
jaccard_null_t20_inc1 = errors_t20_inc1[:,4]
jaccard_input_t20_inc1 = errors_t20_inc1[:,5]

jaccard_t40_inc2 = errors_t40_inc2[:,3]
jaccard_null_t40_inc2 = errors_t40_inc2[:,4]
jaccard_input_t40_inc2 = errors_t40_inc2[:,5]

jaccard_t80_inc4 = errors_t80_inc4[:,3]
jaccard_null_t80_inc4 = errors_t80_inc4[:,4]
jaccard_input_t80_inc4 = errors_t80_inc4[:,5]

data = [jaccard_t20_inc1, jaccard_input_t20_inc1, jaccard_null_t20_inc1]
bplot4 = axs[0,3].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,3].set_title("JACCARD ERRORS\nT=20 TINC=1")
axs[0,3].grid(True)
axs[0,3].set_xticks([])
#axs[0,1].set_ylabel('Error')


data = [jaccard_t40_inc2, jaccard_input_t40_inc2, jaccard_null_t40_inc2]
bplot5 = axs[1,3].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,3].set_title("T=40 TINC=2")
axs[1,3].grid(True)
axs[1,3].set_xticks([])
#axs[1,1].set_ylabel('Error')


data = [jaccard_t80_inc4, jaccard_input_t80_inc4, jaccard_null_t80_inc4]
bplot6 = axs[2,3].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[2,3].set_title("T=80 TINC=4")
axs[2,3].grid(True)
#axs[2,1].set_ylabel('Error')
axs[2,3].set_xticks([])


# fill with colors

for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

fig.legend([bplot1["boxes"][0], bplot1["boxes"][1], bplot1["boxes"][2], bplot1["means"][0]], ["Pred. Seq.(~xt1...~xt20) vs Target Seq.(xt1...xt20)", "Pred. Seq.(~xt1...~xt20) vs Input Seq.(xt0...xt19)", "Target Seq.(xt1...xt20) vs Input Seq.(xt0...xt19) - Null Model", "Mean"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure)



#fig.legend(["Pred. Seq.(~xt1...~xt20) vs Target Seq.(xt1...xt20)", "Pred. Seq.(~xt1...~xt20) vs Input Seq.(xt0...xt19)", "Target Seq.(xt1...xt20) vs Input Seq.(xt0...xt19) - Null Model"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/ConvLSTM3D_MSEJaccard_Errors_Histogram_Boxplot.png"), bbox_inches="tight")