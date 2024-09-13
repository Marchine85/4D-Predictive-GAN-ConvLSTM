import os
import numpy as np
import matplotlib.pyplot as plt
    
base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'plots')):
    os.makedirs(os.path.join(base_dir,'plots'))

# IMPORT SINGLE FRAME INPUT MODEL ERRORS
errors_t20_inc1_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T20_TINC1.txt"), delimiter=",", skiprows=1)
errors_t40_inc2_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T40_TINC2.txt"), delimiter=",", skiprows=1)
errors_t80_inc4_t3 = np.loadtxt(os.path.join(base_dir, "evals\AEncGAN3D_TestSet_Errors_T80_TINC4.txt"), delimiter=",", skiprows=1)

###############################################################################
# PLOT mse ERRORS
###############################################################################
mse_t20_inc1 = errors_t20_inc1_t3[:,0]
mse_t20_inc1_mean = np.round(np.mean(mse_t20_inc1),4)
mse_t20_inc1_stddev = np.round(np.std(mse_t20_inc1),4)
mse_null_t20_inc1 = errors_t20_inc1_t3[:,1]
mse_null_t20_inc1_mean = np.round(np.mean(mse_null_t20_inc1),4)
mse_null_t20_inc1_stddev = np.round(np.std(mse_null_t20_inc1),4)
mse_input_t20_inc1 = errors_t20_inc1_t3[:,2]
mse_input_t20_inc1_mean = np.round(np.mean(mse_input_t20_inc1),4)
mse_input_t20_inc1_stddev = np.round(np.std(mse_input_t20_inc1),4)

mse_t40_inc2 = errors_t40_inc2_t3[:,0]
mse_t40_inc2_mean = np.round(np.mean(mse_t40_inc2),4)
mse_t40_inc2_stddev = np.round(np.std(mse_t40_inc2),4)
mse_null_t40_inc2 = errors_t40_inc2_t3[:,1]
mse_null_t40_inc2_mean = np.round(np.mean(mse_null_t40_inc2),4)
mse_null_t40_inc2_stddev = np.round(np.std(mse_null_t40_inc2),4)
mse_input_t40_inc2 = errors_t40_inc2_t3[:,2]
mse_input_t40_inc2_mean = np.round(np.mean(mse_input_t40_inc2),4)
mse_input_t40_inc2_stddev = np.round(np.std(mse_input_t40_inc2),4)

mse_t80_inc4 = errors_t80_inc4_t3[:,0]
mse_t80_inc4_mean = np.round(np.mean(mse_t80_inc4),4)
mse_t80_inc4_stddev = np.round(np.std(mse_t80_inc4),4)
mse_null_t80_inc4 = errors_t80_inc4_t3[:,1]
mse_null_t80_inc4_mean = np.round(np.mean(mse_null_t80_inc4),4)
mse_null_t80_inc4_stddev = np.round(np.std(mse_null_t80_inc4),4)
mse_input_t80_inc4 = errors_t80_inc4_t3[:,2]
mse_input_t80_inc4_mean = np.round(np.mean(mse_input_t80_inc4),4)
mse_input_t80_inc4_stddev = np.round(np.std(mse_input_t80_inc4),4)

mse_pred = np.concatenate(([[mse_t20_inc1_mean,mse_t20_inc1_stddev]],[[mse_t40_inc2_mean,mse_t40_inc2_stddev]],[[mse_t80_inc4_mean,mse_t80_inc4_stddev]]), axis=1)
mse_input = np.concatenate(([[mse_input_t20_inc1_mean,mse_input_t20_inc1_stddev]],[[mse_input_t40_inc2_mean,mse_input_t40_inc2_stddev]],[[mse_input_t80_inc4_mean,mse_input_t80_inc4_stddev]]), axis=1)
mse_null = np.concatenate(([[mse_null_t20_inc1_mean,mse_null_t20_inc1_stddev]],[[mse_null_t40_inc2_mean,mse_null_t40_inc2_stddev]],[[mse_null_t80_inc4_mean,mse_null_t80_inc4_stddev]]), axis=1)


jaccard_t20_inc1 = errors_t20_inc1_t3[:,3]
jaccard_t20_inc1_mean = np.round(np.mean(jaccard_t20_inc1),4)
jaccard_t20_inc1_stddev = np.round(np.std(jaccard_t20_inc1),4)
jaccard_null_t20_inc1 = errors_t20_inc1_t3[:,4]
jaccard_null_t20_inc1_mean = np.round(np.mean(jaccard_null_t20_inc1),4)
jaccard_null_t20_inc1_stddev = np.round(np.std(jaccard_null_t20_inc1),4)
jaccard_input_t20_inc1 = errors_t20_inc1_t3[:,5]
jaccard_input_t20_inc1_mean = np.round(np.mean(jaccard_input_t20_inc1),4)
jaccard_input_t20_inc1_stddev = np.round(np.std(jaccard_input_t20_inc1),4)

jaccard_t40_inc2 = errors_t40_inc2_t3[:,3]
jaccard_t40_inc2_mean = np.round(np.mean(jaccard_t40_inc2),4)
jaccard_t40_inc2_stddev = np.round(np.std(jaccard_t40_inc2),4)
jaccard_null_t40_inc2 = errors_t40_inc2_t3[:,4]
jaccard_null_t40_inc2_mean = np.round(np.mean(jaccard_null_t40_inc2),4)
jaccard_null_t40_inc2_stddev = np.round(np.std(jaccard_null_t40_inc2),4)
jaccard_input_t40_inc2 = errors_t40_inc2_t3[:,5]
jaccard_input_t40_inc2_mean = np.round(np.mean(jaccard_input_t40_inc2),4)
jaccard_input_t40_inc2_stddev = np.round(np.std(jaccard_input_t40_inc2),4)

jaccard_t80_inc4 = errors_t80_inc4_t3[:,3]
jaccard_t80_inc4_mean = np.round(np.mean(jaccard_t80_inc4),4)
jaccard_t80_inc4_stddev = np.round(np.std(jaccard_t80_inc4),4)
jaccard_null_t80_inc4 = errors_t80_inc4_t3[:,4]
jaccard_null_t80_inc4_mean = np.round(np.mean(jaccard_null_t80_inc4),4)
jaccard_null_t80_inc4_stddev = np.round(np.std(jaccard_null_t80_inc4),4)
jaccard_input_t80_inc4 = errors_t80_inc4_t3[:,5]
jaccard_input_t80_inc4_mean = np.round(np.mean(jaccard_input_t80_inc4),4)
jaccard_input_t80_inc4_stddev = np.round(np.std(jaccard_input_t80_inc4),4)

jaccard_pred = np.concatenate(([[jaccard_t20_inc1_mean,jaccard_t20_inc1_stddev]],[[jaccard_t40_inc2_mean,jaccard_t40_inc2_stddev]],[[jaccard_t80_inc4_mean,jaccard_t80_inc4_stddev]]), axis=1)
jaccard_input = np.concatenate(([[jaccard_input_t20_inc1_mean,jaccard_input_t20_inc1_stddev]],[[jaccard_input_t40_inc2_mean,jaccard_input_t40_inc2_stddev]],[[jaccard_input_t80_inc4_mean,jaccard_input_t80_inc4_stddev]]), axis=1)
jaccard_null = np.concatenate(([[jaccard_null_t20_inc1_mean,jaccard_null_t20_inc1_stddev]],[[jaccard_null_t40_inc2_mean,jaccard_null_t40_inc2_stddev]],[[jaccard_null_t80_inc4_mean,jaccard_null_t80_inc4_stddev]]), axis=1)



res_name = np.array([["pred_mse"],["input_mse"],["null_mse"],["pred_jaccard"],["input_jaccard"],["null_jaccard"]], dtype=str)

all_results = np.concatenate((mse_pred,mse_input,mse_null,jaccard_pred,jaccard_input,jaccard_null), axis=0)

all_results = np.concatenate((res_name,all_results), axis=1)
header = "            Output,      Mean T20 TINC1,    Stddev T20 TINC1,      Mean T40 TINC2,    Stddev T40 TINC2,      Mean T80 TINC4,    Stddev T80 TINC4"
np.savetxt(f"evals/AEncGAN3D_TestSet_MeanStddev_Single.txt" , all_results, fmt="%20s", delimiter=',', header=header)
