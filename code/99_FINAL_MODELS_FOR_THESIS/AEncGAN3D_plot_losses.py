import os
import numpy as np
import matplotlib.pyplot as plt

    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

###############################################################################
# PLOT GENERATOR AND DISCRIMINATOR TRAINING LOSSES
###############################################################################

# IMPORT SINGLE FRAME INPUT MODEL TRAINING LOSSES
aencgan3d_t20_inc1_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T20_TINC1\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t20_inc1_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T20_TINC1\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T20_TINC1 = aencgan3d_t20_inc1_train_Losses[:,1]
D_Train_Loss_T20_TINC1 = aencgan3d_t20_inc1_train_Losses[:,2]
D_Val_Loss_T20_TINC1 = aencgan3d_t20_inc1_val_Losses[:,1]

aencgan3d_t40_inc2_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T40_TINC2\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t40_inc2_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T40_TINC2\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T40_TINC2 = aencgan3d_t40_inc2_train_Losses[:,1]
D_Train_Loss_T40_TINC2 = aencgan3d_t40_inc2_train_Losses[:,2]
D_Val_Loss_T40_TINC2 = aencgan3d_t40_inc2_val_Losses[:,1]

aencgan3d_t80_inc4_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T80_TINC4\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t80_inc4_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T80_TINC4\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T80_TINC4 = aencgan3d_t80_inc4_train_Losses[:,1]
D_Train_Loss_T80_TINC4 = aencgan3d_t80_inc4_train_Losses[:,2]
D_Val_Loss_T80_TINC4 = aencgan3d_t80_inc4_val_Losses[:,1]

# IMPORT MULTIPLE FRAME INPUT MODEL TRAINING LOSSES
aencgan3d_t20_inc1_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T20_TINC1\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t20_inc1_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T20_TINC1\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_train_Losses[:,1]
D_Train_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_train_Losses[:,2]
D_Val_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_val_Losses[:,1]

aencgan3d_t40_inc2_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T40_TINC2\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t40_inc2_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T40_TINC2\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_train_Losses[:,1]
D_Train_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_train_Losses[:,2]
D_Val_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_val_Losses[:,1]

aencgan3d_t80_inc4_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t80_inc4_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4\plots\_val_losses.txt"), delimiter=",", skiprows=1)
G_Train_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_train_Losses[:,1]
D_Train_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_train_Losses[:,2]
D_Val_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_val_Losses[:,1]

iters = aencgan3d_t20_inc1_train_Losses[:,0]
val_iters = aencgan3d_t20_inc1_val_Losses[:,0]

# PLOT
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, layout="constrained", figsize=(12, 6))

a, = axs[0, 0].plot(iters, G_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='G-Train-Loss')
b, = axs[0, 0].plot(iters, D_Train_Loss_T20_TINC1, color='navy', alpha=0.5, label='D-Train-Loss')
c, = axs[0, 0].plot(val_iters, D_Val_Loss_T20_TINC1, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[0, 0].set_title("Single Frame Input Model\nT=20 TINC=1")
axs[0, 0].grid(True)
#axs[0, 0].legend(loc="upper left") 
axs[0, 0].set_ylabel('Loss')

axs[1, 0].plot(iters, G_Train_Loss_T40_TINC2, color='crimson', alpha=0.5, label='G-Train-Loss')
axs[1, 0].plot(iters, D_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='D-Train-Loss')
axs[1, 0].plot(val_iters, D_Val_Loss_T40_TINC2, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[1, 0].set_title("T=40 TINC=2")
axs[1, 0].grid(True)
#axs[1, 0].legend(loc="upper left") 
axs[1, 0].set_ylabel('Loss')

axs[2, 0].plot(iters, G_Train_Loss_T80_TINC4, color='crimson', alpha=0.5, label='G-Train-Loss')
axs[2, 0].plot(iters, D_Train_Loss_T80_TINC4, color='navy', alpha=0.5, label='D-Train-Loss')
axs[2, 0].plot(val_iters, D_Val_Loss_T80_TINC4, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[2, 0].set_title("T=80 TINC=4")
axs[2, 0].grid(True)
#axs[2, 0].legend(loc="upper left") 
axs[2, 0].set_xlabel('Training Iteration (processed batches)')
#axs[2, 0].set_ylabel('Loss')

axs[0, 1].plot(iters, G_Train_Loss_T20_TINC1_t3, color='crimson', alpha=0.5, label='G-Train-Loss')
axs[0, 1].plot(iters, D_Train_Loss_T20_TINC1_t3, color='navy', alpha=0.5, label='D-Train-Loss')
axs[0, 1].plot(val_iters, D_Val_Loss_T20_TINC1_t3, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[0, 1].set_title("Multiple Frame Input Model\nT=20 TINC=1")
axs[0, 1].grid(True)
#axs[0, 1].legend(loc="upper left") 

axs[1, 1].plot(iters, G_Train_Loss_T40_TINC2_t3, color='crimson', alpha=0.5, label='G-Train-Loss')
axs[1, 1].plot(iters, D_Train_Loss_T40_TINC2_t3, color='navy', alpha=0.5, label='D-Train-Loss')
axs[1, 1].plot(val_iters, D_Val_Loss_T40_TINC2_t3, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[1, 1].set_title("T=40 TINC=2")
axs[1, 1].grid(True)
#axs[1, 1].legend(loc="upper left") 

axs[2, 1].plot(iters, G_Train_Loss_T80_TINC4_t3, color='crimson', alpha=0.5, label='G-Train-Loss')
axs[2, 1].plot(iters, D_Train_Loss_T80_TINC4_t3, color='navy', alpha=0.5, label='D-Train-Loss')
axs[2, 1].plot(val_iters, D_Val_Loss_T80_TINC4_t3, color='navy', linestyle='dashed', alpha=0.5, label='D-Val-Loss')
axs[2, 1].set_title("T=80 TINC=4")
axs[2, 1].grid(True)
#axs[2, 1].legend(loc="upper left") 
axs[2, 1].set_xlabel('Training Iteration (processed batches)')

plt.ylim([-50, np.max([G_Train_Loss_T20_TINC1, D_Train_Loss_T20_TINC1, G_Train_Loss_T40_TINC2, D_Train_Loss_T40_TINC2, G_Train_Loss_T80_TINC4, D_Train_Loss_T80_TINC4])*1.1])

fig.legend([a, b, c], ["G-Train-Loss", "D-Train-Loss", "D-Val-Loss"], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.07), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/AEncGAN3D_GD_losses.png"), bbox_inches="tight")

# ###############################################################################
# # PLOT MSE AND JACCARD TRAINING LOSSES
# ###############################################################################

# IMPORT SINGLE FRAME INPUT MODEL TRAINING LOSSES
aencgan3d_t20_inc1_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T20_TINC1\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t20_inc1_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T20_TINC1\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T20_TINC1 = aencgan3d_t20_inc1_train_Losses[:,3]
JACCARD_Train_Loss_T20_TINC1 = aencgan3d_t20_inc1_train_Losses[:,4]
MSE_Val_Loss_T20_TINC1 = aencgan3d_t20_inc1_val_Losses[:,2]
JACCARD_Val_Loss_T20_TINC1 = aencgan3d_t20_inc1_val_Losses[:,3]

aencgan3d_t40_inc2_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T40_TINC2\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t40_inc2_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T40_TINC2\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T40_TINC2 = aencgan3d_t40_inc2_train_Losses[:,3]
JACCARD_Train_Loss_T40_TINC2 = aencgan3d_t40_inc2_train_Losses[:,4]
MSE_Val_Loss_T40_TINC2 = aencgan3d_t40_inc2_val_Losses[:,2]
JACCARD_Val_Loss_T40_TINC2 = aencgan3d_t40_inc2_val_Losses[:,3]

aencgan3d_t80_inc4_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T80_TINC4\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t80_inc4_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_T80_TINC4\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T80_TINC4 = aencgan3d_t80_inc4_train_Losses[:,3]
JACCARD_Train_Loss_T80_TINC4 = aencgan3d_t80_inc4_train_Losses[:,4]
MSE_Val_Loss_T80_TINC4 = aencgan3d_t80_inc4_val_Losses[:,2]
JACCARD_Val_Loss_T80_TINC4 = aencgan3d_t80_inc4_val_Losses[:,3]

# IMPORT MULTIPLE FRAME INPUT MODEL TRAINING LOSSES
aencgan3d_t20_inc1_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T20_TINC1\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t20_inc1_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T20_TINC1\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_train_Losses[:,3]
JACCARD_Train_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_train_Losses[:,4]
MSE_Val_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_val_Losses[:,2]
JACCARD_Val_Loss_T20_TINC1_t3 = aencgan3d_t20_inc1_t3_val_Losses[:,3]

aencgan3d_t40_inc2_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T40_TINC2\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t40_inc2_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T40_TINC2\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_train_Losses[:,3]
JACCARD_Train_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_train_Losses[:,4]
MSE_Val_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_val_Losses[:,2]
JACCARD_Val_Loss_T40_TINC2_t3 = aencgan3d_t40_inc2_t3_val_Losses[:,3]

aencgan3d_t80_inc4_t3_train_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan3d_t80_inc4_t3_val_Losses = np.loadtxt(os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4\plots\_val_losses.txt"), delimiter=",", skiprows=1)
MSE_Train_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_train_Losses[:,3]
JACCARD_Train_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_train_Losses[:,4]
MSE_Val_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_val_Losses[:,2]
JACCARD_Val_Loss_T80_TINC4_t3 = aencgan3d_t80_inc4_t3_val_Losses[:,3]

iters = aencgan3d_t20_inc1_train_Losses[:,0]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, layout="constrained", figsize=(12, 4))

a, = axs[0, 0].plot(iters, MSE_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='MSE-Train-Loss T=20 TINC=1')
b, = axs[0, 0].plot(iters, MSE_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='MSE-Train-Loss T=40 TINC=2')
c, = axs[0, 0].plot(iters, MSE_Train_Loss_T80_TINC4, color='seagreen', alpha=0.5, label='MSE-Train-Loss T=80 TINC=4')
d, = axs[0, 0].plot(val_iters, MSE_Val_Loss_T20_TINC1, color='crimson', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=20 TINC=1')
e, = axs[0, 0].plot(val_iters, MSE_Val_Loss_T40_TINC2, color='navy', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=40 TINC=2')
f, = axs[0, 0].plot(val_iters, MSE_Val_Loss_T80_TINC4, color='seagreen', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=80 TINC=4')
axs[0, 0].set_title("Single Frame Input Model\nMSE LOSS")
axs[0, 0].grid(True)
#axs[0, 0].legend(loc="upper right") 
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_ylim([0, 0.2])

axs[1, 0].plot(iters, JACCARD_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='JACCARD-Train-Loss T=20 TINC=1')
axs[1, 0].plot(iters, JACCARD_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='JACCARD-Train-Loss T=40 TINC=2')
axs[1, 0].plot(iters, JACCARD_Train_Loss_T80_TINC4, color='seagreen', alpha=0.5, label='JACCARD-Train-Loss T=80 TINC=4')
axs[1, 0].plot(val_iters, JACCARD_Val_Loss_T20_TINC1, color='crimson', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=20 TINC=1')
axs[1, 0].plot(val_iters, JACCARD_Val_Loss_T40_TINC2, color='navy', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=40 TINC=2')
axs[1, 0].plot(val_iters, JACCARD_Val_Loss_T80_TINC4, color='seagreen', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=80 TINC=4')
axs[1, 0].set_title("JACCARD LOSS")
axs[1, 0].grid(True)
#axs[1, 0].legend(loc="upper right") 
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_ylim([0, 1.0])
axs[1, 0].set_xlabel('Training Iteration (processed batches)')


axs[0, 1].plot(iters, MSE_Train_Loss_T20_TINC1_t3, color='crimson', alpha=0.5, label='MSE-Train-Loss T=20 TINC=1')
axs[0, 1].plot(iters, MSE_Train_Loss_T40_TINC2_t3, color='navy', alpha=0.5, label='MSE-Train-Loss T=40 TINC=2')
axs[0, 1].plot(iters, MSE_Train_Loss_T80_TINC4_t3, color='seagreen', alpha=0.5, label='MSE-Train-Loss T=80 TINC=4')
axs[0, 1].plot(val_iters, MSE_Val_Loss_T20_TINC1_t3, color='crimson', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=20 TINC=1')
axs[0, 1].plot(val_iters, MSE_Val_Loss_T40_TINC2_t3, color='navy', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=40 TINC=2')
axs[0, 1].plot(val_iters, MSE_Val_Loss_T80_TINC4_t3, color='seagreen', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=80 TINC=4')
axs[0, 1].set_title("Multiple Frame Input Model\nMSE LOSS")
axs[0, 1].grid(True)
#axs[0, 1].legend(loc="upper right") 
axs[0, 1].set_ylabel('Loss')
axs[0, 1].set_ylim([0, 0.2])

axs[1, 1].plot(iters, JACCARD_Train_Loss_T20_TINC1_t3, color='crimson', alpha=0.5, label='JACCARD-Train-Loss T=20 TINC=1')
axs[1, 1].plot(iters, JACCARD_Train_Loss_T40_TINC2_t3, color='navy', alpha=0.5, label='JACCARD-Train-Loss T=40 TINC=2')
axs[1, 1].plot(iters, JACCARD_Train_Loss_T80_TINC4_t3, color='seagreen', alpha=0.5, label='JACCARD-Train-Loss T=80 TINC=4')
axs[1, 1].plot(val_iters, JACCARD_Val_Loss_T20_TINC1_t3, color='crimson', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=20 TINC=1')
axs[1, 1].plot(val_iters, JACCARD_Val_Loss_T40_TINC2_t3, color='navy', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=40 TINC=2')
axs[1, 1].plot(val_iters, JACCARD_Val_Loss_T80_TINC4_t3, color='seagreen', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=80 TINC=4')
axs[1, 1].set_title("JACCARD LOSS")
axs[1, 1].grid(True)
#axs[1, 1].legend(loc="upper right") 
axs[1, 1].set_ylabel('Loss')
axs[1, 1].set_ylim([0, 1.0])
axs[1, 1].set_xlabel('Training Iteration (processed batches)')

fig.legend([a, d, b, e, c, f], ["Train-Loss T=20 TINC=1", "Val-Loss T=20 TINC=1", "Train-Loss T=40 TINC=2", "Val-Loss T=40 TINC=2", "Train-Loss T=80 TINC=4", "Val-Loss T=80 TINC=4"], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.15), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/AEncGAN3D_MSE_JACCARD_losses.png"), bbox_inches="tight")

