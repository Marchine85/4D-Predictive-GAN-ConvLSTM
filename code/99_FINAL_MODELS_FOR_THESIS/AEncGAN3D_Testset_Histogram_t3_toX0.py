import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT Multiple FRAME INPUT MODEL ERRORS
errors_t20_inc1_t3_toX0 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T20_TINC1_t3_toX0.txt"), delimiter=",", skiprows=1)
errors_t40_inc2_t3_toX0 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T40_TINC2_t3_toX0.txt"), delimiter=",", skiprows=1)
errors_t80_inc4_t3_toX0 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T80_TINC4_t3_toX0.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT MSE ERRORS
###############################################################################
mse_t20_inc1 = errors_t20_inc1_t3_toX0[:,0]
mse_t20_inc1_mean = np.round(np.mean(mse_t20_inc1),4)
mse_t20_inc1_stddev = np.round(np.std(mse_t20_inc1),4)
mse_null_t20_inc1 = errors_t20_inc1_t3_toX0[:,1]
mse_null_t20_inc1_mean = np.round(np.mean(mse_null_t20_inc1),4)
mse_null_t20_inc1_stddev = np.round(np.std(mse_null_t20_inc1),4)
mse_input_t20_inc1 = errors_t20_inc1_t3_toX0[:,2]
mse_input_t20_inc1_mean = np.round(np.mean(mse_input_t20_inc1),4)
mse_input_t20_inc1_stddev = np.round(np.std(mse_input_t20_inc1),4)

mse_t40_inc2 = errors_t40_inc2_t3_toX0[:,0]
mse_t40_inc2_mean = np.round(np.mean(mse_t40_inc2),4)
mse_t40_inc2_stddev = np.round(np.std(mse_t40_inc2),4)
mse_null_t40_inc2 = errors_t40_inc2_t3_toX0[:,1]
mse_null_t40_inc2_mean = np.round(np.mean(mse_null_t40_inc2),4)
mse_null_t40_inc2_stddev = np.round(np.std(mse_null_t40_inc2),4)
mse_input_t40_inc2 = errors_t40_inc2_t3_toX0[:,2]
mse_input_t40_inc2_mean = np.round(np.mean(mse_input_t40_inc2),4)
mse_input_t40_inc2_stddev = np.round(np.std(mse_input_t40_inc2),4)

mse_t80_inc4 = errors_t80_inc4_t3_toX0[:,0]
mse_t80_inc4_mean = np.round(np.mean(mse_t80_inc4),4)
mse_t80_inc4_stddev = np.round(np.std(mse_t80_inc4),4)
mse_null_t80_inc4 = errors_t80_inc4_t3_toX0[:,1]
mse_null_t80_inc4_mean = np.round(np.mean(mse_null_t80_inc4),4)
mse_null_t80_inc4_stddev = np.round(np.std(mse_null_t80_inc4),4)
mse_input_t80_inc4 = errors_t80_inc4_t3_toX0[:,2]
mse_input_t80_inc4_mean = np.round(np.mean(mse_input_t80_inc4),4)
mse_input_t80_inc4_stddev = np.round(np.std(mse_input_t80_inc4),4)

fig, axs = plt.subplots(3, 2, sharex=False, sharey=True, layout="constrained", figsize=(10, 10))
colors = ["crimson", "navy", "seagreen"]


labels = [f"µ={mse_t20_inc1_mean}, σ={mse_t20_inc1_stddev}", 
          f"µ={mse_input_t20_inc1_mean}, σ={mse_input_t20_inc1_stddev}",
          f"µ={mse_null_t20_inc1_mean}, σ={mse_null_t20_inc1_stddev}"]
data = [mse_t20_inc1, mse_input_t20_inc1, mse_null_t20_inc1]
axs[0,0].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[0,0].set_title("MSE ERRORS - Multiple Frame Input Model\nT=20 TINC=1")
axs[0,0].grid(True)
axs[0,0].legend(loc="upper right") 
axs[0,0].set_ylabel('Count')
axs[0,0].set_xlim([0,0.15])

labels = [f"µ={mse_t40_inc2_mean}, σ={mse_t40_inc2_stddev} ",  
          f"µ={mse_input_t40_inc2_mean}, σ={mse_input_t40_inc2_stddev} ",
          f"µ={mse_null_t40_inc2_mean}, σ={mse_null_t40_inc2_stddev} ",]
data = [mse_t40_inc2, mse_input_t40_inc2, mse_null_t40_inc2]
axs[1,0].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[1,0].set_title("T=40 TINC=2")
axs[1,0].grid(True)
axs[1,0].legend(loc="upper right") 
axs[1,0].set_ylabel('Count')
axs[1,0].set_xlim([0,0.15])

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
axs[2,0].set_xlim([0,0.15])

###############################################################################
# PLOT JACCARD ERRORS
###############################################################################
jaccard_t20_inc1 = errors_t20_inc1_t3_toX0[:,3]
jaccard_t20_inc1_mean = np.round(np.mean(jaccard_t20_inc1),4)
jaccard_t20_inc1_stddev = np.round(np.std(jaccard_t20_inc1),4)
jaccard_null_t20_inc1 = errors_t20_inc1_t3_toX0[:,4]
jaccard_null_t20_inc1_mean = np.round(np.mean(jaccard_null_t20_inc1),4)
jaccard_null_t20_inc1_stddev = np.round(np.std(jaccard_null_t20_inc1),4)
jaccard_input_t20_inc1 = errors_t20_inc1_t3_toX0[:,5]
jaccard_input_t20_inc1_mean = np.round(np.mean(jaccard_input_t20_inc1),4)
jaccard_input_t20_inc1_stddev = np.round(np.std(jaccard_input_t20_inc1),4)

jaccard_t40_inc2 = errors_t40_inc2_t3_toX0[:,3]
jaccard_t40_inc2_mean = np.round(np.mean(jaccard_t40_inc2),4)
jaccard_t40_inc2_stddev = np.round(np.std(jaccard_t40_inc2),4)
jaccard_null_t40_inc2 = errors_t40_inc2_t3_toX0[:,4]
jaccard_null_t40_inc2_mean = np.round(np.mean(jaccard_null_t40_inc2),4)
jaccard_null_t40_inc2_stddev = np.round(np.std(jaccard_null_t40_inc2),4)
jaccard_input_t40_inc2 = errors_t40_inc2_t3_toX0[:,5]
jaccard_input_t40_inc2_mean = np.round(np.mean(jaccard_input_t40_inc2),4)
jaccard_input_t40_inc2_stddev = np.round(np.std(jaccard_input_t40_inc2),4)

jaccard_t80_inc4 = errors_t80_inc4_t3_toX0[:,3]
jaccard_t80_inc4_mean = np.round(np.mean(jaccard_t80_inc4),4)
jaccard_t80_inc4_stddev = np.round(np.std(jaccard_t80_inc4),4)
jaccard_null_t80_inc4 = errors_t80_inc4_t3_toX0[:,4]
jaccard_null_t80_inc4_mean = np.round(np.mean(jaccard_null_t80_inc4),4)
jaccard_null_t80_inc4_stddev = np.round(np.std(jaccard_null_t80_inc4),4)
jaccard_input_t80_inc4 = errors_t80_inc4_t3_toX0[:,5]
jaccard_input_t80_inc4_mean = np.round(np.mean(jaccard_input_t80_inc4),4)
jaccard_input_t80_inc4_stddev = np.round(np.std(jaccard_input_t80_inc4),4)


labels = [f"µ={jaccard_t20_inc1_mean}, σ={jaccard_t20_inc1_stddev}", 
          f"µ={jaccard_input_t20_inc1_mean}, σ={jaccard_input_t20_inc1_stddev}",
          f"µ={jaccard_null_t20_inc1_mean}, σ={jaccard_null_t20_inc1_stddev}"]
data = [jaccard_t20_inc1, jaccard_input_t20_inc1, jaccard_null_t20_inc1]
axs[0,1].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[0,1].set_title("JACCARD ERRORS - Multiple Frame Input Model\nT=20 TINC=1")
axs[0,1].grid(True)
axs[0,1].legend(loc="upper right") 
#axs[0,1].set_ylabel('Count')
axs[0,1].set_xlim([0,1])

labels = [f"µ={jaccard_t40_inc2_mean}, σ={jaccard_t40_inc2_stddev} ",  
          f"µ={jaccard_input_t40_inc2_mean}, σ={jaccard_input_t40_inc2_stddev} ",
          f"µ={jaccard_null_t40_inc2_mean}, σ={jaccard_null_t40_inc2_stddev} ",]
data = [jaccard_t40_inc2, jaccard_input_t40_inc2, jaccard_null_t40_inc2]
axs[1,1].hist(data, bins = 50, histtype='bar', stacked=True, color=colors, alpha=0.5, label=labels)
axs[1,1].set_title("T=40 TINC=2")
axs[1,1].grid(True)
axs[1,1].legend(loc="upper right") 
#axs[1,1].set_ylabel('Count')
axs[1,1].set_xlim([0,1])

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
axs[2,1].set_xlim([0,1])



fig.legend(["Prediction(~xt1) vs Target(xt1)", "Prediction(~xt1) vs Input(xt0)", "Target(xt1) vs Input(xt0) - Null Model"], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.04), bbox_transform=fig.transFigure)


plt.savefig(os.path.join(base_dir,"plots/AEncGAN3D_MSEJaccard_Errors_Histogram_t3_toX0.png"), bbox_inches="tight")