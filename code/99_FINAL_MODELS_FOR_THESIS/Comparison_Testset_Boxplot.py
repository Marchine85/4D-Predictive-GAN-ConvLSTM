import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT SINGLE FRAME INPUT MODEL ERRORS
single_errors_t20_inc1_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T20_TINC1.txt"), delimiter=",", skiprows=1)
single_errors_t40_inc2_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T40_TINC2.txt"), delimiter=",", skiprows=1)
single_errors_t80_inc4_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T80_TINC4.txt"), delimiter=",", skiprows=1)

# IMPORT MULTIPLE FRAME INPUT MODEL ERRORS
multiple_errors_t20_inc1_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T20_TINC1_t3_toX0.txt"), delimiter=",", skiprows=1)
multiple_errors_t40_inc2_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T40_TINC2_t3_toX0.txt"), delimiter=",", skiprows=1)
multiple_errors_t80_inc4_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T80_TINC4_t3_toX0.txt"), delimiter=",", skiprows=1)

# IMPORT CONVLSTM MODEL ERRORS
convlstm_errors_t20_inc1_t3 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T20_TINC1.txt"), delimiter=",", skiprows=1)
convlstm_errors_t40_inc2_t3 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T40_TINC2.txt"), delimiter=",", skiprows=1)
convlstm_errors_t80_inc4_t3 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_T80_TINC4.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT MSE ERRORS
###############################################################################
single_mse_t20_inc1 = single_errors_t20_inc1_t3[:,0]
single_mse_null_t20_inc1 = single_errors_t20_inc1_t3[:,1]
single_mse_input_t20_inc1 = single_errors_t20_inc1_t3[:,2]
multiple_mse_t20_inc1 = multiple_errors_t20_inc1_t3[:,0]
multiple_mse_null_t20_inc1 = multiple_errors_t20_inc1_t3[:,1]
multiple_mse_input_t20_inc1 = multiple_errors_t20_inc1_t3[:,2]
convlstm_mse_t20_inc1 = convlstm_errors_t20_inc1_t3[:,0]
convlstm_mse_null_t20_inc1 = convlstm_errors_t20_inc1_t3[:,1]
convlstm_mse_input_t20_inc1 = convlstm_errors_t20_inc1_t3[:,2]

single_mse_t40_inc2 = single_errors_t40_inc2_t3[:,0]
single_mse_null_t40_inc2 = single_errors_t40_inc2_t3[:,1]
single_mse_input_t40_inc2 = single_errors_t40_inc2_t3[:,2]
multiple_mse_t40_inc2 = multiple_errors_t40_inc2_t3[:,0]
multiple_mse_null_t40_inc2 = multiple_errors_t40_inc2_t3[:,1]
multiple_mse_input_t40_inc2 = multiple_errors_t40_inc2_t3[:,2]
convlstm_mse_t40_inc2 = convlstm_errors_t40_inc2_t3[:,0]
convlstm_mse_null_t40_inc2 = convlstm_errors_t40_inc2_t3[:,1]
convlstm_mse_input_t40_inc2 = convlstm_errors_t40_inc2_t3[:,2]

single_mse_t80_inc4 = single_errors_t80_inc4_t3[:,0]
single_mse_null_t80_inc4 = single_errors_t80_inc4_t3[:,1]
single_mse_input_t80_inc4 = single_errors_t80_inc4_t3[:,2]
multiple_mse_t80_inc4 = multiple_errors_t80_inc4_t3[:,0]
multiple_mse_null_t80_inc4 = multiple_errors_t80_inc4_t3[:,1]
multiple_mse_input_t80_inc4 = multiple_errors_t80_inc4_t3[:,2]
convlstm_mse_t80_inc4 = convlstm_errors_t80_inc4_t3[:,0]
convlstm_mse_null_t80_inc4 = convlstm_errors_t80_inc4_t3[:,1]
convlstm_mse_input_t80_inc4 = convlstm_errors_t80_inc4_t3[:,2]

###############################################################################

mean_mse_null_t20_inc1 = np.mean(single_mse_null_t20_inc1)
stddev_mse_null_t20_inc1 = np.std(single_mse_null_t20_inc1)
single_mean_mse_input_t20_inc1 = np.mean(single_mse_input_t20_inc1)
multiple_mean_mse_input_t20_inc1 = np.mean(multiple_mse_input_t20_inc1)
convlstm_mean_mse_input_t20_inc1 = np.mean(convlstm_mse_input_t20_inc1)

mean_mse_null_t40_inc2 = np.mean(single_mse_null_t40_inc2)
stddev_mse_null_t40_inc2 = np.std(single_mse_null_t40_inc2)
single_mean_mse_input_t40_inc2 = np.mean(single_mse_input_t40_inc2)
multiple_mean_mse_input_t40_inc2 = np.mean(multiple_mse_input_t40_inc2)
convlstm_mean_mse_input_t40_inc2 = np.mean(convlstm_mse_input_t40_inc2)

mean_mse_null_t80_inc4 = np.mean(single_mse_null_t80_inc4)
stddev_mse_null_t80_inc4 = np.std(single_mse_null_t80_inc4)
single_mean_mse_input_t80_inc4 = np.mean(single_mse_input_t80_inc4)
multiple_mean_mse_input_t80_inc4 = np.mean(multiple_mse_input_t80_inc4)
convlstm_mean_mse_input_t80_inc4 = np.mean(convlstm_mse_input_t80_inc4)

###############################################################################

fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, layout="constrained", figsize=(12, 8))
colors = ["crimson", "navy", "seagreen", "crimson", "navy", "seagreen", "crimson", "navy", "seagreen", ]
meanlineprops = dict(linestyle='', linewidth=0.8, color='black', marker="*", markeredgecolor='black', markerfacecolor='black')
medianlineprops = dict(linestyle='-', linewidth=0.8, color='black')

labels = ["T20 TINC1", "T40 TINC2", "T80 TINC4", "T20 TINC1", "T40 TINC2", "T80 TINC4", "T20 TINC1", "T40 TINC2", "T80 TINC4"]

# data = [single_mse_t20_inc1, single_mse_t40_inc2, single_mse_t80_inc4, 
#         multiple_mse_t20_inc1, multiple_mse_t40_inc2, multiple_mse_t80_inc4, 
#         convlstm_mse_t20_inc1, convlstm_mse_t40_inc2, convlstm_mse_t80_inc4]
data = [single_mse_t20_inc1, multiple_mse_t20_inc1, convlstm_mse_t20_inc1, 
        single_mse_t40_inc2, multiple_mse_t40_inc2, convlstm_mse_t40_inc2, 
        single_mse_t80_inc4, multiple_mse_t80_inc4, convlstm_mse_t80_inc4]

bplot1 = axs[0,0].boxplot(data[0:3], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
a, = axs[0,0].plot([0.5,3.5], np.ones(2)*mean_mse_null_t20_inc1, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
b, = axs[0,0].plot([0.5,1.5], np.ones(2)*single_mean_mse_input_t20_inc1, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
c, = axs[0,0].plot([1.5,2.5], np.ones(2)*multiple_mean_mse_input_t20_inc1, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
d, = axs[0,0].plot([2.5,3.5], np.ones(2)*convlstm_mean_mse_input_t20_inc1, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,0].set_title("T=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].set_ylabel('Error')
axs[0,0].set_ylim([-0.001,0.055])
axs[0,0].set_xticklabels([None]*3)

bplot2 = axs[0,1].boxplot(data[3:6], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,1].plot([0.5,3.5], np.ones(2)*mean_mse_null_t40_inc2, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
axs[0,1].plot([0.5,1.5], np.ones(2)*single_mean_mse_input_t40_inc2, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,1].plot([1.5,2.5], np.ones(2)*multiple_mean_mse_input_t40_inc2, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,1].plot([2.5,3.5], np.ones(2)*convlstm_mean_mse_input_t40_inc2, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,1].set_title("MSE ERRORS\nT=40 TINC=2")
axs[0,1].grid(True)
axs[0,1].set_ylim([-0.001,0.055])
axs[0,1].set_xticklabels([None]*3)
axs[0,1].set_yticklabels([])

bplot3 = axs[0,2].boxplot(data[6:9], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[0,2].plot([0.5,3.5], np.ones(2)*mean_mse_null_t80_inc4, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
axs[0,2].plot([0.5,1.5], np.ones(2)*single_mean_mse_input_t80_inc4, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,2].plot([1.5,2.5], np.ones(2)*multiple_mean_mse_input_t80_inc4, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,2].plot([2.5,3.5], np.ones(2)*convlstm_mean_mse_input_t80_inc4, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,2].set_title("T=80 TINC=4")
axs[0,2].grid(True)
axs[0,2].set_ylim([-0.001,0.055])
axs[0,2].set_xticklabels([None]*3)
axs[0,2].set_yticklabels([])

# ###############################################################################
# # PLOT JACCARD ERRORS
# ###############################################################################
single_jaccard_t20_inc1 = single_errors_t20_inc1_t3[:,3]
single_jaccard_null_t20_inc1 = single_errors_t20_inc1_t3[:,4]
single_jaccard_input_t20_inc1 = single_errors_t20_inc1_t3[:,5]
multiple_jaccard_t20_inc1 = multiple_errors_t20_inc1_t3[:,3]
multiple_jaccard_null_t20_inc1 = multiple_errors_t20_inc1_t3[:,4]
multiple_jaccard_input_t20_inc1 = multiple_errors_t20_inc1_t3[:,5]
convlstm_jaccard_t20_inc1 = convlstm_errors_t20_inc1_t3[:,3]
convlstm_jaccard_null_t20_inc1 = convlstm_errors_t20_inc1_t3[:,4]
convlstm_jaccard_input_t20_inc1 = convlstm_errors_t20_inc1_t3[:,5]

single_jaccard_t40_inc2 = single_errors_t40_inc2_t3[:,3]
single_jaccard_null_t40_inc2 = single_errors_t40_inc2_t3[:,4]
single_jaccard_input_t40_inc2 = single_errors_t40_inc2_t3[:,5]
multiple_jaccard_t40_inc2 = multiple_errors_t40_inc2_t3[:,3]
multiple_jaccard_null_t40_inc2 = multiple_errors_t40_inc2_t3[:,4]
multiple_jaccard_input_t40_inc2 = multiple_errors_t40_inc2_t3[:,5]
convlstm_jaccard_t40_inc2 = convlstm_errors_t40_inc2_t3[:,3]
convlstm_jaccard_null_t40_inc2 = convlstm_errors_t40_inc2_t3[:,4]
convlstm_jaccard_input_t40_inc2 = convlstm_errors_t40_inc2_t3[:,5]

single_jaccard_t80_inc4 = single_errors_t80_inc4_t3[:,3]
single_jaccard_null_t80_inc4 = single_errors_t80_inc4_t3[:,4]
single_jaccard_input_t80_inc4 = single_errors_t80_inc4_t3[:,5]
multiple_jaccard_t80_inc4 = multiple_errors_t80_inc4_t3[:,3]
multiple_jaccard_null_t80_inc4 = multiple_errors_t80_inc4_t3[:,4]
multiple_jaccard_input_t80_inc4 = multiple_errors_t80_inc4_t3[:,5]
convlstm_jaccard_t80_inc4 = convlstm_errors_t80_inc4_t3[:,3]
convlstm_jaccard_null_t80_inc4 = convlstm_errors_t80_inc4_t3[:,4]
convlstm_jaccard_input_t80_inc4 = convlstm_errors_t80_inc4_t3[:,5]

mean_jaccard_null_t20_inc1 = np.mean(single_jaccard_null_t20_inc1)
stddev_jaccard_null_t20_inc1 = np.std(single_jaccard_null_t20_inc1)
single_mean_jaccard_input_t20_inc1 = np.mean(single_jaccard_input_t20_inc1)
multiple_mean_jaccard_input_t20_inc1 = np.mean(multiple_jaccard_input_t20_inc1)
convlstm_mean_jaccard_input_t20_inc1 = np.mean(convlstm_jaccard_input_t20_inc1)

mean_jaccard_null_t40_inc2 = np.mean(single_jaccard_null_t40_inc2)
stddev_jaccard_null_t40_inc2 = np.std(single_jaccard_null_t40_inc2)
single_mean_jaccard_input_t40_inc2 = np.mean(single_jaccard_input_t40_inc2)
multiple_mean_jaccard_input_t40_inc2 = np.mean(multiple_jaccard_input_t40_inc2)
convlstm_mean_jaccard_input_t40_inc2 = np.mean(convlstm_jaccard_input_t40_inc2)

mean_jaccard_null_t80_inc4 = np.mean(single_jaccard_null_t80_inc4)
stddev_jaccard_null_t80_inc4 = np.std(single_jaccard_null_t80_inc4)
single_mean_jaccard_input_t80_inc4 = np.mean(single_jaccard_input_t80_inc4)
multiple_mean_jaccard_input_t80_inc4 = np.mean(multiple_jaccard_input_t80_inc4)
convlstm_mean_jaccard_input_t80_inc4 = np.mean(convlstm_jaccard_input_t80_inc4)

# data = [single_jaccard_t20_inc1, single_jaccard_t40_inc2, single_jaccard_t80_inc4, 
#         multiple_jaccard_t20_inc1, multiple_jaccard_t40_inc2, multiple_jaccard_t80_inc4, 
#         convlstm_jaccard_t20_inc1, convlstm_jaccard_t40_inc2, convlstm_jaccard_t80_inc4]
data = [single_jaccard_t20_inc1, multiple_jaccard_t20_inc1, convlstm_jaccard_t20_inc1, 
        single_jaccard_t40_inc2, multiple_jaccard_t40_inc2, convlstm_jaccard_t40_inc2, 
        single_jaccard_t80_inc4, multiple_jaccard_t80_inc4, convlstm_jaccard_t80_inc4]

bplot4 = axs[1,0].boxplot(data[0:3], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,0].plot([0.5,3.5], np.ones(2)*mean_jaccard_null_t20_inc1, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
axs[1,0].plot([0.5,1.5], np.ones(2)*single_mean_jaccard_input_t20_inc1, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,0].plot([1.5,2.5], np.ones(2)*multiple_mean_jaccard_input_t20_inc1, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,0].plot([2.5,3.5], np.ones(2)*convlstm_mean_jaccard_input_t20_inc1, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,0].set_title("T=20 TINC=1")
axs[1,0].grid(True)
axs[1,0].set_ylabel('Error')
axs[1,0].set_ylim([-0.02,0.75])
axs[1,0].set_xticklabels([None]*3)

bplot5 = axs[1,1].boxplot(data[3:6], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,1].plot([0.5,3.5], np.ones(2)*mean_jaccard_null_t40_inc2, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
axs[1,1].plot([0.5,1.5], np.ones(2)*single_mean_jaccard_input_t40_inc2, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,1].plot([1.5,2.5], np.ones(2)*multiple_mean_jaccard_input_t40_inc2, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,1].plot([2.5,3.5], np.ones(2)*convlstm_mean_jaccard_input_t40_inc2, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,1].set_title("JACCARD ERRORS\nT=40 TINC=2")
axs[1,1].grid(True)
axs[1,1].set_ylim([-0.02,0.75])
axs[1,1].set_xticklabels([None]*3)
axs[1,1].set_yticklabels([])

bplot6 = axs[1,2].boxplot(data[6:9], vert=True, patch_artist=True, showfliers=False, showmeans=True, meanline=False, meanprops=meanlineprops, medianprops=medianlineprops)
axs[1,2].plot([0.5,3.5], np.ones(2)*mean_jaccard_null_t80_inc4, color='black',  linewidth=1.5, alpha=1.0, linestyle='-.' )
axs[1,2].plot([0.5,1.5], np.ones(2)*single_mean_jaccard_input_t80_inc4, color='crimson',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,2].plot([1.5,2.5], np.ones(2)*multiple_mean_jaccard_input_t80_inc4, color='navy',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,2].plot([2.5,3.5], np.ones(2)*convlstm_mean_jaccard_input_t80_inc4, color='seagreen',  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,2].set_title("T=80 TINC=4")
axs[1,2].grid(True)
axs[1,2].set_ylim([-0.02,0.75])
axs[1,2].set_xticklabels([None]*3)
axs[1,2].set_yticklabels([])


# fill with colors

for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

fig.legend([bplot1["boxes"][0], bplot1["boxes"][1], bplot1["boxes"][2], a, b, c, d, bplot1["means"][0]], ["Single Frame Input Model", "Multiple Frame input Model", "ConvLSTM Model", "µ(Target(xt1) vs Input(xt0)) - Null Model", "Single Frame Input Model µ(Prediction(~xt1) vs Input(xt0))", "Multiple Frame Input Model µ(Prediction(~xt1) vs Input(xt0))", "ConvLSTM Model µ(Prediction(~xt1) vs Input(xt0))", "µ(Prediction(~xt1) vs Target(xt1))"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.13), bbox_transform=fig.transFigure)


plt.savefig(os.path.join(base_dir,"plots/Comparison_MSEJaccard_Errors_Boxplot.png"), bbox_inches="tight")