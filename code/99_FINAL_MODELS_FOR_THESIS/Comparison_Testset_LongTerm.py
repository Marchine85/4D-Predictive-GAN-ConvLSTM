import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT SINGLE FRAME INPUT MODEL ERRORS
single_errors_t20_inc1 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T20_TINC1.txt"), delimiter=",", skiprows=1)
single_errors_t40_inc2 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T40_TINC2.txt"), delimiter=",", skiprows=1)
single_errors_t80_inc4 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T80_TINC4.txt"), delimiter=",", skiprows=1)

# IMPORT MULTIPLE FRAME INPUT MODEL ERRORS
multiple_errors_t20_inc1 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T20_TINC1_t3.txt"), delimiter=",", skiprows=1)
multiple_errors_t40_inc2 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T40_TINC2_t3.txt"), delimiter=",", skiprows=1)
multiple_errors_t80_inc4 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_LongTerm_T80_TINC4_t3.txt"), delimiter=",", skiprows=1)

# IMPORT CONVLSTM MODEL ERRORS
convlstm_errors_t20_inc1 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T20_TINC1.txt"), delimiter=",", skiprows=1)
convlstm_errors_t40_inc2 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T40_TINC2.txt"), delimiter=",", skiprows=1)
convlstm_errors_t80_inc4 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T80_TINC4.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT MSE ERRORS
###############################################################################
single_mse_t20_inc1 = single_errors_t20_inc1[:,0:5]
single_mse_null_t20_inc1 = single_errors_t20_inc1[:,5:10]
single_mse_input_t20_inc1 = single_errors_t20_inc1[:,10:15]
multiple_mse_t20_inc1 = multiple_errors_t20_inc1[:,0:5]
multiple_mse_null_t20_inc1 = multiple_errors_t20_inc1[:,5:10]
multiple_mse_input_t20_inc1 = multiple_errors_t20_inc1[:,10:15]
convlstm_mse_t20_inc1 = convlstm_errors_t20_inc1[:,0:5]
convlstm_mse_null_t20_inc1 = convlstm_errors_t20_inc1[:,5:10]
convlstm_mse_input_t20_inc1 = convlstm_errors_t20_inc1[:,10:15]

single_mse_t40_inc2 = single_errors_t40_inc2[:,0:5]
single_mse_null_t40_inc2 = single_errors_t40_inc2[:,5:10]
single_mse_input_t40_inc2 = single_errors_t40_inc2[:,10:15]
multiple_mse_t40_inc2 = multiple_errors_t40_inc2[:,0:5]
multiple_mse_null_t40_inc2 = multiple_errors_t40_inc2[:,5:10]
multiple_mse_input_t40_inc2 = multiple_errors_t40_inc2[:,10:15]
convlstm_mse_t40_inc2 = convlstm_errors_t40_inc2[:,0:5]
convlstm_mse_null_t40_inc2 = convlstm_errors_t40_inc2[:,5:10]
convlstm_mse_input_t40_inc2 = convlstm_errors_t40_inc2[:,10:15]

single_mse_t80_inc4 = single_errors_t80_inc4[:,0:5]
single_mse_null_t80_inc4 = single_errors_t80_inc4[:,5:10]
single_mse_input_t80_inc4 = single_errors_t80_inc4[:,10:15]
multiple_mse_t80_inc4 = multiple_errors_t80_inc4[:,0:5]
multiple_mse_null_t80_inc4 = multiple_errors_t80_inc4[:,5:10]
multiple_mse_input_t80_inc4 = multiple_errors_t80_inc4[:,10:15]
convlstm_mse_t80_inc4 = convlstm_errors_t80_inc4[:,0:5]
convlstm_mse_null_t80_inc4 = convlstm_errors_t80_inc4[:,5:10]
convlstm_mse_input_t80_inc4 = convlstm_errors_t80_inc4[:,10:15]

###############################################################################

single_mse_t20_inc1_mean = np.mean(single_mse_t20_inc1, axis=0)
multiple_mse_t20_inc1_mean = np.mean(multiple_mse_t20_inc1, axis=0)
convlstm_mse_t20_inc1_mean = np.mean(convlstm_mse_t20_inc1, axis=0)
single_mse_t40_inc2_mean = np.mean(single_mse_t40_inc2, axis=0)
multiple_mse_t40_inc2_mean = np.mean(multiple_mse_t40_inc2, axis=0)
convlstm_mse_t40_inc2_mean = np.mean(convlstm_mse_t40_inc2, axis=0)
single_mse_t80_inc4_mean = np.mean(single_mse_t80_inc4, axis=0)
multiple_mse_t80_inc4_mean = np.mean(multiple_mse_t80_inc4, axis=0)
convlstm_mse_t80_inc4_mean = np.mean(convlstm_mse_t80_inc4, axis=0)

single_mse_input_t20_inc1_mean = np.mean(single_mse_input_t20_inc1, axis=0)
multiple_mse_input_t20_inc1_mean = np.mean(multiple_mse_input_t20_inc1, axis=0)
convlstm_mse_input_t20_inc1_mean = np.mean(convlstm_mse_input_t20_inc1, axis=0)
single_mse_input_t40_inc2_mean = np.mean(single_mse_input_t40_inc2, axis=0)
multiple_mse_input_t40_inc2_mean = np.mean(multiple_mse_input_t40_inc2, axis=0)
convlstm_mse_input_t40_inc2_mean = np.mean(convlstm_mse_input_t40_inc2, axis=0)
single_mse_input_t80_inc4_mean = np.mean(single_mse_input_t80_inc4, axis=0)
multiple_mse_input_t80_inc4_mean = np.mean(multiple_mse_input_t80_inc4, axis=0)
convlstm_mse_input_t80_inc4_mean = np.mean(convlstm_mse_input_t80_inc4, axis=0)

single_mse_null_t20_inc1_mean = np.mean(single_mse_null_t20_inc1, axis=0)
multiple_mse_null_t20_inc1_mean = np.mean(multiple_mse_null_t20_inc1, axis=0)
convlstm_mse_null_t20_inc1_mean = np.mean(convlstm_mse_null_t20_inc1, axis=0)
single_mse_null_t40_inc2_mean = np.mean(single_mse_null_t40_inc2, axis=0)
multiple_mse_null_t40_inc2_mean = np.mean(multiple_mse_null_t40_inc2, axis=0)
convlstm_mse_null_t40_inc2_mean = np.mean(convlstm_mse_null_t40_inc2, axis=0)
single_mse_null_t80_inc4_mean = np.mean(single_mse_null_t80_inc4, axis=0)
multiple_mse_null_t80_inc4_mean = np.mean(multiple_mse_null_t80_inc4, axis=0)
convlstm_mse_null_t80_inc4_mean = np.mean(convlstm_mse_null_t80_inc4, axis=0)

###############################################################################

fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, layout="constrained", figsize=(12, 6))
colors = ["crimson", "navy", "seagreen", "crimson", "navy", "seagreen", "crimson", "navy", "seagreen", ]
meanlineprops = dict(linestyle='', linewidth=0.8, color='black', marker="*", markeredgecolor='black', markerfacecolor='black')
medianlineprops = dict(linestyle='-', linewidth=0.8, color='black')

a, = axs[0,0].plot(np.arange(1,6,1), single_mse_t20_inc1_mean, color="crimson",  linewidth=1.5, alpha=0.5)
b, = axs[0,0].plot(np.arange(1,6,1), multiple_mse_t20_inc1_mean, color="navy",  linewidth=1.5, alpha=0.5)
c, = axs[0,0].plot(np.arange(1,6,1), convlstm_mse_t20_inc1_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
d, = axs[0,0].plot(np.arange(1,6,1), single_mse_input_t20_inc1_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
e, = axs[0,0].plot(np.arange(1,6,1), multiple_mse_input_t20_inc1_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
f, = axs[0,0].plot(np.arange(1,6,1), convlstm_mse_input_t20_inc1_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
g, = axs[0,0].plot(np.arange(1,6,1), convlstm_mse_null_t20_inc1_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[0,0].set_title("T=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].set_ylabel('Error')
axs[0,0].set_ylim([-0.001,0.09])


axs[0,1].plot(np.arange(1,6,1), single_mse_t40_inc2_mean, color="crimson",  linewidth=1.5, alpha=0.5)
axs[0,1].plot(np.arange(1,6,1), multiple_mse_t40_inc2_mean, color="navy",  linewidth=1.5, alpha=0.5)
axs[0,1].plot(np.arange(1,6,1), convlstm_mse_t40_inc2_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
axs[0,1].plot(np.arange(1,6,1), single_mse_input_t40_inc2_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,1].plot(np.arange(1,6,1), multiple_mse_input_t40_inc2_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
axs[0,1].plot(np.arange(1,6,1), convlstm_mse_input_t40_inc2_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,1].plot(np.arange(1,6,1), convlstm_mse_null_t40_inc2_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[0,1].set_title("T=40 TINC=2")
axs[0,1].grid(True)
axs[0,1].set_ylim([-0.001,0.09])
axs[0,1].set_yticklabels([])

axs[0,2].plot(np.arange(1,6,1), single_mse_t80_inc4_mean, color="crimson",  linewidth=1.5, alpha=0.5)
axs[0,2].plot(np.arange(1,6,1), multiple_mse_t80_inc4_mean, color="navy",  linewidth=1.5, alpha=0.5)
axs[0,2].plot(np.arange(1,6,1), convlstm_mse_t80_inc4_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
axs[0,2].plot(np.arange(1,6,1), single_mse_input_t80_inc4_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,2].plot(np.arange(1,6,1), multiple_mse_input_t80_inc4_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
axs[0,2].plot(np.arange(1,6,1), convlstm_mse_input_t80_inc4_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[0,2].plot(np.arange(1,6,1), convlstm_mse_null_t80_inc4_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[0,2].set_title("T=80 TINC=4")
axs[0,2].grid(True)
axs[0,2].set_ylim([-0.001,0.09])
axs[0,2].set_yticklabels([])

# ###############################################################################
# # PLOT JACCARD ERRORS
# ###############################################################################
single_jaccard_t20_inc1 = single_errors_t20_inc1[:,15:20]
single_jaccard_null_t20_inc1 = single_errors_t20_inc1[:,20:25]
single_jaccard_input_t20_inc1 = single_errors_t20_inc1[:,25:30]
multiple_jaccard_t20_inc1 = multiple_errors_t20_inc1[:,15:20]
multiple_jaccard_null_t20_inc1 = multiple_errors_t20_inc1[:,20:25]
multiple_jaccard_input_t20_inc1 = multiple_errors_t20_inc1[:,25:30]
convlstm_jaccard_t20_inc1 = convlstm_errors_t20_inc1[:,15:20]
convlstm_jaccard_null_t20_inc1 = convlstm_errors_t20_inc1[:,20:25]
convlstm_jaccard_input_t20_inc1 = convlstm_errors_t20_inc1[:,25:30]

single_jaccard_t40_inc2 = single_errors_t40_inc2[:,15:20]
single_jaccard_null_t40_inc2 = single_errors_t40_inc2[:,20:25]
single_jaccard_input_t40_inc2 = single_errors_t40_inc2[:,25:30]
multiple_jaccard_t40_inc2 = multiple_errors_t40_inc2[:,15:20]
multiple_jaccard_null_t40_inc2 = multiple_errors_t40_inc2[:,20:25]
multiple_jaccard_input_t40_inc2 = multiple_errors_t40_inc2[:,25:30]
convlstm_jaccard_t40_inc2 = convlstm_errors_t40_inc2[:,15:20]
convlstm_jaccard_null_t40_inc2 = convlstm_errors_t40_inc2[:,20:25]
convlstm_jaccard_input_t40_inc2 = convlstm_errors_t40_inc2[:,25:30]

single_jaccard_t80_inc4 = single_errors_t80_inc4[:,15:20]
single_jaccard_null_t80_inc4 = single_errors_t80_inc4[:,20:25]
single_jaccard_input_t80_inc4 = single_errors_t80_inc4[:,25:30]
multiple_jaccard_t80_inc4 = multiple_errors_t80_inc4[:,15:20]
multiple_jaccard_null_t80_inc4 = multiple_errors_t80_inc4[:,20:25]
multiple_jaccard_input_t80_inc4 = multiple_errors_t80_inc4[:,25:30]
convlstm_jaccard_t80_inc4 = convlstm_errors_t80_inc4[:,15:20]
convlstm_jaccard_null_t80_inc4 = convlstm_errors_t80_inc4[:,20:25]
convlstm_jaccard_input_t80_inc4 = convlstm_errors_t80_inc4[:,25:30]

###############################################################################

single_jaccard_t20_inc1_mean = np.mean(single_jaccard_t20_inc1, axis=0)
multiple_jaccard_t20_inc1_mean = np.mean(multiple_jaccard_t20_inc1, axis=0)
convlstm_jaccard_t20_inc1_mean = np.mean(convlstm_jaccard_t20_inc1, axis=0)
single_jaccard_t40_inc2_mean = np.mean(single_jaccard_t40_inc2, axis=0)
multiple_jaccard_t40_inc2_mean = np.mean(multiple_jaccard_t40_inc2, axis=0)
convlstm_jaccard_t40_inc2_mean = np.mean(convlstm_jaccard_t40_inc2, axis=0)
single_jaccard_t80_inc4_mean = np.mean(single_jaccard_t80_inc4, axis=0)
multiple_jaccard_t80_inc4_mean = np.mean(multiple_jaccard_t80_inc4, axis=0)
convlstm_jaccard_t80_inc4_mean = np.mean(convlstm_jaccard_t80_inc4, axis=0)

single_jaccard_input_t20_inc1_mean = np.mean(single_jaccard_input_t20_inc1, axis=0)
multiple_jaccard_input_t20_inc1_mean = np.mean(multiple_jaccard_input_t20_inc1, axis=0)
convlstm_jaccard_input_t20_inc1_mean = np.mean(convlstm_jaccard_input_t20_inc1, axis=0)
single_jaccard_input_t40_inc2_mean = np.mean(single_jaccard_input_t40_inc2, axis=0)
multiple_jaccard_input_t40_inc2_mean = np.mean(multiple_jaccard_input_t40_inc2, axis=0)
convlstm_jaccard_input_t40_inc2_mean = np.mean(convlstm_jaccard_input_t40_inc2, axis=0)
single_jaccard_input_t80_inc4_mean = np.mean(single_jaccard_input_t80_inc4, axis=0)
multiple_jaccard_input_t80_inc4_mean = np.mean(multiple_jaccard_input_t80_inc4, axis=0)
convlstm_jaccard_input_t80_inc4_mean = np.mean(convlstm_jaccard_input_t80_inc4, axis=0)

single_jaccard_null_t20_inc1_mean = np.mean(single_jaccard_null_t20_inc1, axis=0)
multiple_jaccard_null_t20_inc1_mean = np.mean(multiple_jaccard_null_t20_inc1, axis=0)
convlstm_jaccard_null_t20_inc1_mean = np.mean(convlstm_jaccard_null_t20_inc1, axis=0)
single_jaccard_null_t40_inc2_mean = np.mean(single_jaccard_null_t40_inc2, axis=0)
multiple_jaccard_null_t40_inc2_mean = np.mean(multiple_jaccard_null_t40_inc2, axis=0)
convlstm_jaccard_null_t40_inc2_mean = np.mean(convlstm_jaccard_null_t40_inc2, axis=0)
single_jaccard_null_t80_inc4_mean = np.mean(single_jaccard_null_t80_inc4, axis=0)
multiple_jaccard_null_t80_inc4_mean = np.mean(multiple_jaccard_null_t80_inc4, axis=0)
convlstm_jaccard_null_t80_inc4_mean = np.mean(convlstm_jaccard_null_t80_inc4, axis=0)

###############################################################################

axs[1,0].plot(np.arange(1,6,1), single_jaccard_t20_inc1_mean, color="crimson",  linewidth=1.5, alpha=0.5)
axs[1,0].plot(np.arange(1,6,1), multiple_jaccard_t20_inc1_mean, color="navy",  linewidth=1.5, alpha=0.5)
axs[1,0].plot(np.arange(1,6,1), convlstm_jaccard_t20_inc1_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
axs[1,0].plot(np.arange(1,6,1), single_jaccard_input_t20_inc1_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,0].plot(np.arange(1,6,1), multiple_jaccard_input_t20_inc1_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
axs[1,0].plot(np.arange(1,6,1), convlstm_jaccard_input_t20_inc1_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,0].plot(np.arange(1,6,1), convlstm_jaccard_null_t20_inc1_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[1,0].set_title("T=20 TINC=1")
axs[1,0].grid(True)
axs[1,0].set_ylabel('Error')
axs[1,0].set_ylim([-0.01,1.0])
#axs[1,0].set_xticklabels([None]*5)
axs[1,0].set_xlabel('Time')

axs[1,1].plot(np.arange(1,6,1), single_jaccard_t40_inc2_mean, color="crimson",  linewidth=1.5, alpha=0.5)
axs[1,1].plot(np.arange(1,6,1), multiple_jaccard_t40_inc2_mean, color="navy",  linewidth=1.5, alpha=0.5)
axs[1,1].plot(np.arange(1,6,1), convlstm_jaccard_t40_inc2_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
axs[1,1].plot(np.arange(1,6,1), single_jaccard_input_t40_inc2_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,1].plot(np.arange(1,6,1), multiple_jaccard_input_t40_inc2_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
axs[1,1].plot(np.arange(1,6,1), convlstm_jaccard_input_t40_inc2_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,1].plot(np.arange(1,6,1), convlstm_jaccard_null_t40_inc2_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[1,1].set_title("T=40 TINC=2")
axs[1,1].grid(True)
axs[1,1].set_ylim([-0.01,1.0])
#axs[1,1].set_xticklabels([None]*5)
axs[1,1].set_yticklabels([])
axs[1,1].set_xlabel('Time')

axs[1,2].plot(np.arange(1,6,1), single_jaccard_t80_inc4_mean, color="crimson",  linewidth=1.5, alpha=0.5)
axs[1,2].plot(np.arange(1,6,1), multiple_jaccard_t80_inc4_mean, color="navy",  linewidth=1.5, alpha=0.5)
axs[1,2].plot(np.arange(1,6,1), convlstm_jaccard_t80_inc4_mean, color="seagreen",  linewidth=1.5, alpha=0.5)
axs[1,2].plot(np.arange(1,6,1), single_jaccard_input_t80_inc4_mean, color="crimson",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,2].plot(np.arange(1,6,1), multiple_jaccard_input_t80_inc4_mean, color="navy",  linewidth=1.5, alpha=1.0, linestyle=':'  )
axs[1,2].plot(np.arange(1,6,1), convlstm_jaccard_input_t80_inc4_mean, color="seagreen",  linewidth=1.5, alpha=1.0, linestyle=':' )
axs[1,2].plot(np.arange(1,6,1), convlstm_jaccard_null_t80_inc4_mean, color='black', linewidth=1.5, linestyle='-.' )
axs[1,2].set_title("T=80 TINC=4")
axs[1,2].grid(True)
axs[1,2].set_ylim([-0.01,1.0])
#axs[1,2].set_xticklabels([None]*5)
axs[1,2].set_yticklabels([])
axs[1,2].set_xlabel('Time')

fig.legend([a, b, c, g, d, e, f], ["Single Frame Input Model µ(Prediction(~xt1) vs Target(xt1))", "Multiple Frame input Model µ(Prediction(~xt1) vs Target(xt1))", "ConvLSTM Model µ(Prediction(~xt1) vs Target(xt1))", "µ(Target(xt1) vs Input(xt0)) - Null Model", "Single Frame Input Model µ(Prediction(~xt1) vs Input(xt0))", "Multiple Frame Input Model µ(Prediction(~xt1) vs Input(xt0))", "ConvLSTM Model µ(Prediction(~xt1) vs Input(xt0))"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.16), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/Comparison_MSEJaccard_Errors_LongTerm.png"), bbox_inches="tight")