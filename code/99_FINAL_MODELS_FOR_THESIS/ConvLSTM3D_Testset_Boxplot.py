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
mse_null_t20_inc1 = errors_t20_inc1[:,1]
mse_input_t20_inc1 = errors_t20_inc1[:,2]

mse_t40_inc2 = errors_t40_inc2[:,0]
mse_null_t40_inc2 = errors_t40_inc2[:,1]
mse_input_t40_inc2 = errors_t40_inc2[:,2]

mse_t80_inc4 = errors_t80_inc4[:,0]
mse_null_t80_inc4 = errors_t80_inc4[:,1]
mse_input_t80_inc4 = errors_t80_inc4[:,2]

fig, axs = plt.subplots(3, 2, sharex=True, sharey=False, layout="constrained", figsize=(10, 10))
colors = ["crimson", "navy", "seagreen"]
meanlineprops = dict(linestyle='', linewidth=0.8, color='black', marker="*", markeredgecolor='black', markerfacecolor='black')
medianlineprops = dict(linestyle='-', linewidth=0.8, color='black')

labels = ["Prediction vs Target(xt1)", "Prediction vs Input(xt0)", "Null Model (xt1 vs xt0)"]

data = [mse_t20_inc1, mse_input_t20_inc1, mse_null_t20_inc1]
bplot1 = axs[0,0].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,0].set_title("MSE ERRORS - Single Frame Input Model\nT=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].set_ylabel('Error')
#axs[0,0].set_ylim([0,0.05])


data = [mse_t40_inc2, mse_input_t40_inc2, mse_null_t40_inc2]
bplot2 = axs[1,0].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,0].set_title("T=40 TINC=2")
axs[1,0].grid(True)
axs[1,0].set_ylabel('Error')
#axs[1,0].set_ylim([0,0.05])


data = [mse_t80_inc4, mse_input_t80_inc4, mse_null_t80_inc4]
bplot3 = axs[2,0].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[2,0].set_title("T=80 TINC=4")
axs[2,0].grid(True)
axs[2,0].set_ylabel('Error')
axs[2,0].set_xticks([])
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
bplot4 = axs[0,1].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,1].set_title("JACCARD ERRORS - Single Frame Input Model\nT=20 TINC=1")
axs[0,1].grid(True)
#axs[0,1].set_ylabel('Error')


data = [jaccard_t40_inc2, jaccard_input_t40_inc2, jaccard_null_t40_inc2]
bplot5 = axs[1,1].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,1].set_title("T=40 TINC=2")
axs[1,1].grid(True)
#axs[1,1].set_ylabel('Error')


data = [jaccard_t80_inc4, jaccard_input_t80_inc4, jaccard_null_t80_inc4]
bplot6 = axs[2,1].boxplot(data, vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[2,1].set_title("T=80 TINC=4")
axs[2,1].grid(True)
#axs[2,1].set_ylabel('Error')
axs[2,0].set_xticks([])


# fill with colors

for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

fig.legend([bplot1["boxes"][0], bplot1["boxes"][1], bplot1["boxes"][2], bplot1["means"][0]], ["Pred. Seq.(~xt1...~xt20) vs Target Seq.(xt1...xt20)", "Pred. Seq.(~xt1...~xt20) vs Input Seq.(xt0...xt19)", "Target Seq.(xt1...xt20) vs Input Seq.(xt0...xt19) - Null Model", "Mean"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure)


plt.savefig(os.path.join(base_dir,"plots/ConvLSTM3D_MSEJaccard_Errors_Boxplot.png"), bbox_inches="tight")