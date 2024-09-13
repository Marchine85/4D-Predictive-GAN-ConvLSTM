#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:55:50 2021

@author: mgoldgruber
"""
import os

import tensorflow as tf
from keras.models import load_model
import numpy as np
import time
# OWN LIBRARIES
from utils import dataIO as d 
from utils import parse_tfrecords_4D_16_for_LongTerm_prediction as p4d


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'evals')):
    os.makedirs(os.path.join(base_dir,'evals'))

MODEL_LOCATION      = os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4")

TEST_DATASET_LOC   = "Z:/Master_Thesis/data/02_4D_test_datasets"

T                   = int(MODEL_LOCATION[-8:-6])              
TINC                = int(MODEL_LOCATION[-1])                      
N_TEST  	        = 1000
BATCH_SIZE          = 32
MAX_PREDICT         = 5
###############################################################################
############# GET Data/Objects ################################################
###############################################################################
TEST_FILENAMES = tf.io.gfile.glob(TEST_DATASET_LOC+"/*.tfrecords")[:N_TEST]
test_set_size = len(TEST_FILENAMES)

print()
print("Total Number of Samples for Test:", test_set_size)
print()

# Convert/Decode dataset from tfrecords file for training
test_objects = p4d.get_dataset(TEST_FILENAMES, shuffle=True, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=False)

encoder_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*encoder*.h5"))
encoder = load_model(encoder_model[-1])

generator_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*generator*.h5"))
generator = load_model(generator_model[-1])



def MSE_reconstruction_loss(real, fake):    
    squ = tf.square(real - fake)
    MSE_loss = tf.reduce_mean(squ, axis=[1,2,3])

    return MSE_loss

def JACCARD_reconstruction_loss(real, fake):    
    j_and = tf.math.reduce_sum(np.double(np.bitwise_and(real, fake)), axis=[1,2,3])
    j_or = tf.math.reduce_sum(np.double(np.bitwise_or(real, fake)), axis=[1,2,3])

    JACCARD_loss = np.nan_to_num(1.0 - j_and / j_or, nan=0.0)

    return JACCARD_loss

def evaluate_LONGTERM_predictions(objects):
    # Random Sample at every VAL_SAVE_FREQ
    MSE_LOSSES_all = []
    JACCARD_LOSSES_all = []
    MSE_LOSSES_all_NULL = []
    JACCARD_LOSSES_all_NULL = []
    MSE_LOSSES_all_INPUT = []
    JACCARD_LOSSES_all_INPUT = []
    
    for X in objects:
        X = tf.dtypes.cast(X, tf.float32)

        MSE_LOSSES_per_batch = []
        JACCARD_LOSSES_per_batch = []
        MSE_LOSSES_per_batch_NULL = []
        JACCARD_LOSSES_per_batch_NULL = []
        MSE_LOSSES_per_batch_INPUT = []
        JACCARD_LOSSES_per_batch_INPUT = []

        for t in range(2,X.shape[1]-3):
            X1 = X[:,t+1]

            if t==2:
                X_minus2 = X[:,0]
                X_minus1 = X[:,1]
                X0 = X[:,2]

            # Encode & Generate
            latent_code_tminus2 = encoder(X_minus2)
            latent_code_tminus1 = encoder(X_minus1)
            latent_code_t0 = encoder(X0)
            latent_code = tf.concat([latent_code_tminus2, latent_code_tminus1, latent_code_t0], axis=1)
            pred_object = np.squeeze(generator(latent_code, training=False))
            
            # JACCARD FOR PREDICTIONS
            MSE_LOSS_per_timepoint = MSE_reconstruction_loss(X1, pred_object).numpy()          
            real = np.asarray(X1>0.5, bool)
            fake = np.asarray(pred_object>0.5, bool) 
            JACCARD_LOSS_per_timepoint = JACCARD_reconstruction_loss(real, fake)
        
            # JACCARD AGAINST NULL MODEL (=MODEL THAT WILL PREDICT THE INPUT TIME FRAME EXACTLY)
            MSE_LOSS_per_timepoint_NULL = MSE_reconstruction_loss(X1, X[:,t]).numpy()
            real = np.asarray(X1>0.5, bool)
            fake = np.asarray(X[:,t]>0.5, bool) 
            JACCARD_LOSS_per_timepoint_NULL = JACCARD_reconstruction_loss(real, fake)
        
            # JACCARD AGAINST INPUT FRAME (=JACCARD BETWEEN PRED AND X0)
            MSE_LOSS_per_timepoint_INPUT = MSE_reconstruction_loss(X[:,t], pred_object).numpy()
            real = np.asarray(X[:,t]>0.5, bool)
            fake = np.asarray(pred_object>0.5, bool) 
            JACCARD_LOSS_per_timepoint_INPUT = JACCARD_reconstruction_loss(real, fake)

            
            MSE_LOSSES_per_batch.append(MSE_LOSS_per_timepoint)
            JACCARD_LOSSES_per_batch.append(JACCARD_LOSS_per_timepoint)
            MSE_LOSSES_per_batch_NULL.append(MSE_LOSS_per_timepoint_NULL)
            JACCARD_LOSSES_per_batch_NULL.append(JACCARD_LOSS_per_timepoint_NULL)
            MSE_LOSSES_per_batch_INPUT.append(MSE_LOSS_per_timepoint_INPUT)
            JACCARD_LOSSES_per_batch_INPUT.append(JACCARD_LOSS_per_timepoint_INPUT)

            if t==MAX_PREDICT+2-1:
                break
            
            X_minus2 = X_minus1
            X_minus1 = X0
            X0 = pred_object
            
        MSE_LOSSES_all.append(np.asarray(MSE_LOSSES_per_batch).T)
        JACCARD_LOSSES_all.append(np.asarray(JACCARD_LOSSES_per_batch).T)
        MSE_LOSSES_all_NULL.append(np.asarray(MSE_LOSSES_per_batch_NULL).T)
        JACCARD_LOSSES_all_NULL.append(np.asarray(JACCARD_LOSSES_per_batch_NULL).T)
        MSE_LOSSES_all_INPUT.append(np.asarray(MSE_LOSSES_per_batch_INPUT).T)
        JACCARD_LOSSES_all_INPUT.append(np.asarray(JACCARD_LOSSES_per_batch_INPUT).T)


    JACCARD_LOSSES_all_mean = np.mean(np.asarray(JACCARD_LOSSES_all).reshape((np.asarray(JACCARD_LOSSES_all).shape[0]*np.asarray(JACCARD_LOSSES_all).shape[1],np.asarray(JACCARD_LOSSES_all).shape[2])), axis=0)
    MSE_LOSSES_all_mean = np.mean(np.asarray(MSE_LOSSES_all).reshape((np.asarray(MSE_LOSSES_all).shape[0]*np.asarray(MSE_LOSSES_all).shape[1],np.asarray(MSE_LOSSES_all).shape[2])), axis=0)
    print("MEAN LOSS OVER ALL SAMPLES: ", np.round(JACCARD_LOSSES_all_mean,3), "% | MSE ", np.round(MSE_LOSSES_all_mean,3))

    return np.array(MSE_LOSSES_all), np.array(MSE_LOSSES_all_NULL), np.array(MSE_LOSSES_all_INPUT), np.array(JACCARD_LOSSES_all), np.array(JACCARD_LOSSES_all_NULL), np.array(JACCARD_LOSSES_all_INPUT)


MSE_LOSSES_all, MSE_LOSSES_all_NULL, MSE_LOSSES_all_INPUT, JACCARD_LOSSES_all, JACCARD_LOSSES_all_NULL, JACCARD_LOSSES_all_INPUT = evaluate_LONGTERM_predictions(test_objects)

MSE_LOSSES_all = np.reshape(MSE_LOSSES_all, (MSE_LOSSES_all.shape[0]*MSE_LOSSES_all.shape[1],MAX_PREDICT))
JACCARD_LOSSES_all = np.reshape(JACCARD_LOSSES_all, (JACCARD_LOSSES_all.shape[0]*JACCARD_LOSSES_all.shape[1],MAX_PREDICT))
MSE_LOSSES_all_NULL = np.reshape(MSE_LOSSES_all_NULL, (MSE_LOSSES_all_NULL.shape[0]*MSE_LOSSES_all_NULL.shape[1],MAX_PREDICT))
JACCARD_LOSSES_all_NULL = np.reshape(JACCARD_LOSSES_all_NULL, (JACCARD_LOSSES_all_NULL.shape[0]*JACCARD_LOSSES_all_NULL.shape[1],MAX_PREDICT))
MSE_LOSSES_all_INPUT= np.reshape(MSE_LOSSES_all_INPUT, (MSE_LOSSES_all_INPUT.shape[0]*MSE_LOSSES_all_INPUT.shape[1],MAX_PREDICT))
JACCARD_LOSSES_all_INPUT = np.reshape(JACCARD_LOSSES_all_INPUT, (JACCARD_LOSSES_all_INPUT.shape[0]*JACCARD_LOSSES_all_INPUT.shape[1],MAX_PREDICT))

LOSSES = np.concatenate((MSE_LOSSES_all, MSE_LOSSES_all_NULL, MSE_LOSSES_all_INPUT, JACCARD_LOSSES_all, JACCARD_LOSSES_all_NULL, JACCARD_LOSSES_all_INPUT), axis=1)

header = "       MSE_LOSS_t1,          MSE_LOSS_t2,          MSE_LOSS_t3,          MSE_LOSS_t4,          MSE_LOSS_t5,\
     MSE_LOSS_NULL_t1,     MSE_LOSS_NULL_t2,     MSE_LOSS_NULL_t3,     MSE_LOSS_NULL_t4,     MSE_LOSS_NULL_t5,\
     MSE_LOSS_INPUT_t1,    MSE_LOSS_INPUT_t2,    MSE_LOSS_INPUT_t3,    MSE_LOSS_INPUT_t4,    MSE_LOSS_INPUT_t5,\
     JACCARD_LOSS_t1,      JACCARD_LOSS_t2,      JACCARD_LOSS_t3,      JACCARD_LOSS_t4,      JACCARD_LOSS_t5,\
     JACCARD_LOSS_NULL_t1, JACCARD_LOSS_NULL_t2, JACCARD_LOSS_NULL_t3, JACCARD_LOSS_NULL_t4, JACCARD_LOSS_NULL_t5,\
     JACCARD_LOSS_INPUT_t1, JACCARD_LOSS_INPUT_t2, JACCARD_LOSS_INPUT_t3, JACCARD_LOSS_INPUT_t4, JACCARD_LOSS_INPUT_t5"

np.savetxt(f"evals/AEncGAN3D_TestSet_Errors_LongTerm_T{T}_TINC{TINC}_t3.txt" , LOSSES, fmt="%20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f", delimiter=',', header=header)

