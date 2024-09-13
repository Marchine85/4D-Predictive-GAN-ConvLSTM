import os
import numpy as np
import matplotlib.pyplot as plt

    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# ###############################################################################
# # PLOT BCE, MSE AND JACCARD TRAINING LOSSES
# ###############################################################################

# IMPORT SINGLE FRAME INPUT MODEL TRAINING LOSSES
aencgan_t20_inc1_train_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T20_TINC1\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan_t20_inc1_val_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T20_TINC1\plots\_val_losses.txt"), delimiter=",", skiprows=1)
BCE_Train_Loss_T20_TINC1 = aencgan_t20_inc1_train_Losses[:,1]
MSE_Train_Loss_T20_TINC1 = aencgan_t20_inc1_train_Losses[:,2]
JACCARD_Train_Loss_T20_TINC1 = aencgan_t20_inc1_train_Losses[:,3]
BCE_Val_Loss_T20_TINC1 = aencgan_t20_inc1_val_Losses[:,1]
MSE_Val_Loss_T20_TINC1 = aencgan_t20_inc1_val_Losses[:,2]
JACCARD_Val_Loss_T20_TINC1 = aencgan_t20_inc1_val_Losses[:,3]

aencgan_t40_inc2_train_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T40_TINC2\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan_t40_inc2_val_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T40_TINC2\plots\_val_losses.txt"), delimiter=",", skiprows=1)
BCE_Train_Loss_T40_TINC2 = aencgan_t40_inc2_train_Losses[:,1]
MSE_Train_Loss_T40_TINC2 = aencgan_t40_inc2_train_Losses[:,2]
JACCARD_Train_Loss_T40_TINC2 = aencgan_t40_inc2_train_Losses[:,3]
BCE_Val_Loss_T40_TINC2 = aencgan_t40_inc2_val_Losses[:,1]
MSE_Val_Loss_T40_TINC2 = aencgan_t40_inc2_val_Losses[:,2]
JACCARD_Val_Loss_T40_TINC2 = aencgan_t40_inc2_val_Losses[:,3]

aencgan_t80_inc4_train_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T80_TINC4\plots\_train_losses.txt"), delimiter=",", skiprows=1)
aencgan_t80_inc4_val_Losses = np.loadtxt(os.path.join(base_dir, "convlstm3d_16_T80_TINC4\plots\_val_losses.txt"), delimiter=",", skiprows=1)
BCE_Train_Loss_T80_TINC4 = aencgan_t80_inc4_train_Losses[:,1]
MSE_Train_Loss_T80_TINC4 = aencgan_t80_inc4_train_Losses[:,2]
JACCARD_Train_Loss_T80_TINC4 = aencgan_t80_inc4_train_Losses[:,3]
BCE_Val_Loss_T80_TINC4 = aencgan_t80_inc4_val_Losses[:,1]
MSE_Val_Loss_T80_TINC4 = aencgan_t80_inc4_val_Losses[:,2]
JACCARD_Val_Loss_T80_TINC4 = aencgan_t80_inc4_val_Losses[:,3]


iters = aencgan_t20_inc1_train_Losses[:,0]
val_iters = aencgan_t20_inc1_val_Losses[:,0]

fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, layout="constrained", figsize=(12, 6))

a, = axs[0].plot(iters, BCE_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='MSE-Train-Loss T=20 TINC=1')
b, = axs[0].plot(iters, BCE_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='MSE-Train-Loss T=40 TINC=2')
c, = axs[0].plot(iters, BCE_Train_Loss_T80_TINC4, color='seagreen', alpha=0.5, label='MSE-Train-Loss T=80 TINC=4')
d, = axs[0].plot(val_iters, BCE_Val_Loss_T20_TINC1, color='crimson', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=20 TINC=1')
e, = axs[0].plot(val_iters, BCE_Val_Loss_T40_TINC2, color='navy', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=40 TINC=2')
f, = axs[0].plot(val_iters, BCE_Val_Loss_T80_TINC4, color='seagreen', alpha=0.5, linestyle='dashed', label='MSE-Val-Loss T=80 TINC=4')
axs[0].set_title("BCE LOSS")
axs[0].grid(True)
#axs[0].legend(loc="upper right") 
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 0.1])

axs[1].plot(iters, MSE_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='JACCARD-Train-Loss T=20 TINC=1')
axs[1].plot(iters, MSE_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='JACCARD-Train-Loss T=40 TINC=2')
axs[1].plot(iters, MSE_Train_Loss_T80_TINC4, color='seagreen', alpha=0.5, label='JACCARD-Train-Loss T=80 TINC=4')
axs[1].plot(val_iters, MSE_Val_Loss_T20_TINC1, color='crimson', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=20 TINC=1')
axs[1].plot(val_iters, MSE_Val_Loss_T40_TINC2, color='navy', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=40 TINC=2')
axs[1].plot(val_iters, MSE_Val_Loss_T80_TINC4, color='seagreen', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=80 TINC=4')
axs[1].set_title("MSE LOSS")
axs[1].grid(True)
#axs[1].legend(loc="upper right") 
axs[1].set_ylabel('Loss')
axs[1].set_ylim([0, 0.05])

axs[2].plot(iters, JACCARD_Train_Loss_T20_TINC1, color='crimson', alpha=0.5, label='JACCARD-Train-Loss T=20 TINC=1')
axs[2].plot(iters, JACCARD_Train_Loss_T40_TINC2, color='navy', alpha=0.5, label='JACCARD-Train-Loss T=40 TINC=2')
axs[2].plot(iters, JACCARD_Train_Loss_T80_TINC4, color='seagreen', alpha=0.5, label='JACCARD-Train-Loss T=80 TINC=4')
axs[2].plot(val_iters, JACCARD_Val_Loss_T20_TINC1, color='crimson', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=20 TINC=1')
axs[2].plot(val_iters, JACCARD_Val_Loss_T40_TINC2, color='navy', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=40 TINC=2')
axs[2].plot(val_iters, JACCARD_Val_Loss_T80_TINC4, color='seagreen', alpha=0.5, linestyle='dashed', label='JACCARD-Val-Loss T=80 TINC=4')
axs[2].set_title("JACCARD LOSS")
axs[2].grid(True)
#axs[2].legend(loc="upper right") 
axs[2].set_ylabel('Loss')
axs[2].set_ylim([0, 1.0])
axs[2].set_xlabel('Training Iteration (processed batches)')


fig.legend([a, d, b, e, c, f], ["Train-Loss T=20 TINC=1", "Val-Loss T=20 TINC=1", "Train-Loss T=40 TINC=2", "Val-Loss T=40 TINC=2", "Train-Loss T=80 TINC=4", "Val-Loss T=80 TINC=4"], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.1), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(base_dir,"plots/ConvLSTM3D_BCE_MSE_JACCARD_losses.png"), bbox_inches="tight")

