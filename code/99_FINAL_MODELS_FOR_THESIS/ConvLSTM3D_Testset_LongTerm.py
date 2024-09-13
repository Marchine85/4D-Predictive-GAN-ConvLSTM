import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT SINGLE FRAME INPUT MODEL ERRORS
longterm_t20_inc1 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T20_TINC1.txt"), delimiter=",", skiprows=1)
longterm_t40_inc2 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T40_TINC2.txt"), delimiter=",", skiprows=1)
longterm_t80_inc4 = np.loadtxt(os.path.join(base_dir, "evals\ConvLSTM3D_TestSet_Errors_LongTerm_T80_TINC4.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT MSE ERRORS
###############################################################################
mse_t20_inc1 = longterm_t20_inc1[:,0:5]
mse_t20_inc1_mean = np.round(np.mean(mse_t20_inc1, axis=0),4)
mse_t20_inc1_stddev = np.round(np.std(mse_t20_inc1, axis=0),4)
mse_null_t20_inc1 = longterm_t20_inc1[:,5:10]
mse_null_t20_inc1_mean = np.round(np.mean(mse_null_t20_inc1, axis=0),4)
mse_null_t20_inc1_stddev = np.round(np.std(mse_null_t20_inc1, axis=0),4)
mse_input_t20_inc1 = longterm_t20_inc1[:,10:15]
mse_input_t20_inc1_mean = np.round(np.mean(mse_input_t20_inc1, axis=0),4)
mse_input_t20_inc1_stddev = np.round(np.std(mse_input_t20_inc1, axis=0),4)

mse_t40_inc2 = longterm_t40_inc2[:,0:5]
mse_t40_inc2_mean = np.round(np.mean(mse_t40_inc2, axis=0),4)
mse_t40_inc2_stddev = np.round(np.std(mse_t40_inc2, axis=0),4)
mse_null_t40_inc2 = longterm_t40_inc2[:,5:10]
mse_null_t40_inc2_mean = np.round(np.mean(mse_null_t40_inc2, axis=0),4)
mse_null_t40_inc2_stddev = np.round(np.std(mse_null_t40_inc2, axis=0),4)
mse_input_t40_inc2 = longterm_t40_inc2[:,10:15]
mse_input_t40_inc2_mean = np.round(np.mean(mse_input_t40_inc2, axis=0),4)
mse_input_t40_inc2_stddev = np.round(np.std(mse_input_t40_inc2, axis=0),4)

mse_t80_inc4 = longterm_t80_inc4[:,0:5]
mse_t80_inc4_mean = np.round(np.mean(mse_t80_inc4, axis=0),4)
mse_t80_inc4_stddev = np.round(np.std(mse_t80_inc4, axis=0),4)
mse_null_t80_inc4 = longterm_t80_inc4[:,5:10]
mse_null_t80_inc4_mean = np.round(np.mean(mse_null_t80_inc4, axis=0),4)
mse_null_t80_inc4_stddev = np.round(np.std(mse_null_t80_inc4, axis=0),4)
mse_input_t80_inc4 = longterm_t80_inc4[:,10:15]
mse_input_t80_inc4_mean = np.round(np.mean(mse_input_t80_inc4, axis=0),4)
mse_input_t80_inc4_stddev = np.round(np.std(mse_input_t80_inc4, axis=0),4)

fig, axs = plt.subplots(3, 2, sharex=True, sharey=False, layout="constrained", figsize=(12, 8))
plt.xticks(np.arange(1,6,1))


axs[0,0].plot(np.arange(1,6,1), mse_t20_inc1_mean, color="crimson", alpha=0.5)
axs[0,0].fill_between(np.arange(1,6,1), mse_t20_inc1_mean-mse_t20_inc1_stddev, mse_t20_inc1_mean+mse_t20_inc1_stddev, color="crimson", alpha=0.1)
axs[0,0].plot(np.arange(1,6,1), mse_input_t20_inc1_mean, color="navy", alpha=0.5)
axs[0,0].fill_between(np.arange(1,6,1), mse_input_t20_inc1_mean-mse_input_t20_inc1_stddev, mse_input_t20_inc1_mean+mse_input_t20_inc1_stddev, color="navy", alpha=0.1)
axs[0,0].plot(np.arange(1,6,1), mse_null_t20_inc1_mean, color="seagreen", alpha=0.5)
axs[0,0].fill_between(np.arange(1,6,1), mse_null_t20_inc1_mean-mse_null_t20_inc1_stddev, mse_null_t20_inc1_mean+mse_null_t20_inc1_stddev, color="seagreen", alpha=0.1)
axs[0,0].set_title("MSE ERRORS - ConvLSTM Model\nT=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].set_ylabel('Error')
axs[0,0].set_ylim([-0.005,0.12])

axs[1,0].plot(np.arange(1,6,1), mse_t40_inc2_mean, color="crimson", alpha=0.5)
axs[1,0].fill_between(np.arange(1,6,1), mse_t40_inc2_mean-mse_t40_inc2_stddev, mse_t40_inc2_mean+mse_t40_inc2_stddev, color="crimson", alpha=0.1)
axs[1,0].plot(np.arange(1,6,1), mse_input_t40_inc2_mean, color="navy", alpha=0.5)
axs[1,0].fill_between(np.arange(1,6,1), mse_input_t40_inc2_mean-mse_input_t40_inc2_stddev, mse_input_t40_inc2_mean+mse_input_t40_inc2_stddev, color="navy", alpha=0.1)
axs[1,0].plot(np.arange(1,6,1), mse_null_t40_inc2_mean, color="seagreen", alpha=0.5)
axs[1,0].fill_between(np.arange(1,6,1), mse_null_t40_inc2_mean-mse_null_t40_inc2_stddev, mse_null_t40_inc2_mean+mse_null_t40_inc2_stddev, color="seagreen", alpha=0.1)
axs[1,0].set_title("T=40 TINC=2")
axs[1,0].grid(True)
axs[1,0].set_ylabel('Error')
axs[1,0].set_ylim([-0.005,0.12])

axs[2,0].plot(np.arange(1,6,1), mse_t80_inc4_mean, color="crimson", alpha=0.5)
axs[2,0].fill_between(np.arange(1,6,1), mse_t80_inc4_mean-mse_t80_inc4_stddev, mse_t80_inc4_mean+mse_t80_inc4_stddev, color="crimson", alpha=0.1)
axs[2,0].plot(np.arange(1,6,1), mse_input_t80_inc4_mean, color="navy", alpha=0.5)
axs[2,0].fill_between(np.arange(1,6,1), mse_input_t80_inc4_mean-mse_input_t80_inc4_stddev, mse_input_t80_inc4_mean+mse_input_t80_inc4_stddev, color="navy", alpha=0.1)
axs[2,0].plot(np.arange(1,6,1), mse_null_t80_inc4_mean, color="seagreen", alpha=0.5)
axs[2,0].fill_between(np.arange(1,6,1), mse_null_t80_inc4_mean-mse_null_t80_inc4_stddev, mse_null_t80_inc4_mean+mse_null_t80_inc4_stddev, color="seagreen", alpha=0.1)
axs[2,0].set_title("T=80 TINC=4")
axs[2,0].grid(True)
axs[2,0].set_ylabel('Error')
axs[2,0].set_xlabel('Time')
axs[2,0].set_ylim([-0.005,0.12])


###############################################################################
# PLOT JACCARD ERRORS
###############################################################################
jaccard_t20_inc1 = longterm_t20_inc1[:,15:20]
jaccard_t20_inc1_mean = np.round(np.mean(jaccard_t20_inc1, axis=0),4)
jaccard_t20_inc1_stddev = np.round(np.std(jaccard_t20_inc1, axis=0),4)
jaccard_null_t20_inc1 = longterm_t20_inc1[:,20:25]
jaccard_null_t20_inc1_mean = np.round(np.mean(jaccard_null_t20_inc1, axis=0),4)
jaccard_null_t20_inc1_stddev = np.round(np.std(jaccard_null_t20_inc1, axis=0),4)
jaccard_input_t20_inc1 = longterm_t20_inc1[:,25:30]
jaccard_input_t20_inc1_mean = np.round(np.mean(jaccard_input_t20_inc1, axis=0),4)
jaccard_input_t20_inc1_stddev = np.round(np.std(jaccard_input_t20_inc1, axis=0),4)

jaccard_t40_inc2 = longterm_t40_inc2[:,15:20]
jaccard_t40_inc2_mean = np.round(np.mean(jaccard_t40_inc2, axis=0),4)
jaccard_t40_inc2_stddev = np.round(np.std(jaccard_t40_inc2, axis=0),4)
jaccard_null_t40_inc2 = longterm_t40_inc2[:,20:25]
jaccard_null_t40_inc2_mean = np.round(np.mean(jaccard_null_t40_inc2, axis=0),4)
jaccard_null_t40_inc2_stddev = np.round(np.std(jaccard_null_t40_inc2, axis=0),4)
jaccard_input_t40_inc2 = longterm_t40_inc2[:,25:30]
jaccard_input_t40_inc2_mean = np.round(np.mean(jaccard_input_t40_inc2, axis=0),4)
jaccard_input_t40_inc2_stddev = np.round(np.std(jaccard_input_t40_inc2, axis=0),4)

jaccard_t80_inc4 = longterm_t80_inc4[:,15:20]
jaccard_t80_inc4_mean = np.round(np.mean(jaccard_t80_inc4, axis=0),4)
jaccard_t80_inc4_stddev = np.round(np.std(jaccard_t80_inc4, axis=0),4)
jaccard_null_t80_inc4 = longterm_t80_inc4[:,20:25]
jaccard_null_t80_inc4_mean = np.round(np.mean(jaccard_null_t80_inc4, axis=0),4)
jaccard_null_t80_inc4_stddev = np.round(np.std(jaccard_null_t80_inc4, axis=0),4)
jaccard_input_t80_inc4 = longterm_t80_inc4[:,25:30]
jaccard_input_t80_inc4_mean = np.round(np.mean(jaccard_input_t80_inc4, axis=0),4)
jaccard_input_t80_inc4_stddev = np.round(np.std(jaccard_input_t80_inc4, axis=0),4)


axs[0,1].plot(np.arange(1,6,1), jaccard_t20_inc1_mean, color="crimson", alpha=0.5)
axs[0,1].fill_between(np.arange(1,6,1), jaccard_t20_inc1_mean-jaccard_t20_inc1_stddev, jaccard_t20_inc1_mean+jaccard_t20_inc1_stddev, color="crimson", alpha=0.1)
axs[0,1].plot(np.arange(1,6,1), jaccard_input_t20_inc1_mean, color="navy", alpha=0.5)
axs[0,1].fill_between(np.arange(1,6,1), jaccard_input_t20_inc1_mean-jaccard_input_t20_inc1_stddev, jaccard_input_t20_inc1_mean+jaccard_input_t20_inc1_stddev, color="navy", alpha=0.1)
axs[0,1].plot(np.arange(1,6,1), jaccard_null_t20_inc1_mean, color="seagreen", alpha=0.5)
axs[0,1].fill_between(np.arange(1,6,1), jaccard_null_t20_inc1_mean-jaccard_null_t20_inc1_stddev, jaccard_null_t20_inc1_mean+jaccard_null_t20_inc1_stddev, color="seagreen", alpha=0.1)
axs[0,1].set_title("JACCARD ERRORS - ConvLSTM Model\nT=20 TINC=1")
axs[0,1].grid(True)
axs[0,1].set_ylabel('Error')
axs[0,1].set_ylim([-0.05,1.0])

axs[1,1].plot(np.arange(1,6,1), jaccard_t40_inc2_mean, color="crimson", alpha=0.5)
axs[1,1].fill_between(np.arange(1,6,1), jaccard_t40_inc2_mean-jaccard_t40_inc2_stddev, jaccard_t40_inc2_mean+jaccard_t40_inc2_stddev, color="crimson", alpha=0.1)
axs[1,1].plot(np.arange(1,6,1), jaccard_input_t40_inc2_mean, color="navy", alpha=0.5)
axs[1,1].fill_between(np.arange(1,6,1), jaccard_input_t40_inc2_mean-jaccard_input_t40_inc2_stddev, jaccard_input_t40_inc2_mean+jaccard_input_t40_inc2_stddev, color="navy", alpha=0.1)
axs[1,1].plot(np.arange(1,6,1), jaccard_null_t40_inc2_mean, color="seagreen", alpha=0.5)
axs[1,1].fill_between(np.arange(1,6,1), jaccard_null_t40_inc2_mean-jaccard_null_t40_inc2_stddev, jaccard_null_t40_inc2_mean+jaccard_null_t40_inc2_stddev, color="seagreen", alpha=0.1)
axs[1,1].set_title("T=40 TINC=2")
axs[1,1].grid(True)
axs[1,1].set_ylabel('Error')
axs[1,1].set_ylim([-0.05,1.0])

axs[2,1].plot(np.arange(1,6,1), jaccard_t80_inc4_mean, color="crimson", alpha=0.5)
axs[2,1].fill_between(np.arange(1,6,1), jaccard_t80_inc4_mean-jaccard_t80_inc4_stddev, jaccard_t80_inc4_mean+jaccard_t80_inc4_stddev, color="crimson", alpha=0.1)
axs[2,1].plot(np.arange(1,6,1), jaccard_input_t80_inc4_mean, color="navy", alpha=0.5)
axs[2,1].fill_between(np.arange(1,6,1), jaccard_input_t80_inc4_mean-jaccard_input_t80_inc4_stddev, jaccard_input_t80_inc4_mean+jaccard_input_t80_inc4_stddev, color="navy", alpha=0.1)
axs[2,1].plot(np.arange(1,6,1), jaccard_null_t80_inc4_mean, color="seagreen", alpha=0.5)
axs[2,1].fill_between(np.arange(1,6,1), jaccard_null_t80_inc4_mean-jaccard_null_t80_inc4_stddev, jaccard_null_t80_inc4_mean+jaccard_null_t80_inc4_stddev, color="seagreen", alpha=0.1)
axs[2,1].set_title("T=80 TINC=4")
axs[2,1].grid(True)
axs[2,1].set_ylabel('Error')
axs[2,1].set_xlabel('Time')
axs[2,1].set_ylim([-0.05,1.0])


fig.legend(["µ(Prediction(~xt1) vs Target(xt1))", "σ(Prediction(~xt1) vs Target(xt1))", "µ(Prediction(~xt1) vs Input(xt0))", "σ(Prediction(~xt1) vs Input(xt0))",  "µ(Target(xt1) vs Input(xt0)) - Null Model",  "σ(Target(xt1) vs Input(xt0)) - Null Model"], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/ConvLSTM3D_LongTerm_MSEJaccard_Errors.png"), bbox_inches="tight")