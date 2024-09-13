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
from utils import parse_tfrecords_4D_16_for_training as p4d


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'evals')):
    os.makedirs(os.path.join(base_dir,'evals'))

MODEL_LOCATION      = os.path.join(base_dir, "aencgan3d_16_T80_TINC4")

TEST_DATASET_LOC   = "Z:/Master_Thesis/data/02_4D_test_datasets"

T                   = int(MODEL_LOCATION[-8:-6])              
TINC                = int(MODEL_LOCATION[-1])                      
N_TEST  	        = 1000
BATCH_SIZE          = 32
###############################################################################
############# GET Data/Objects ################################################
###############################################################################
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_DATASET_LOC,"*.tfrecords"))[:N_TEST]
test_set_size = len(TEST_FILENAMES)
test_samples = test_set_size * (T//TINC-1) // BATCH_SIZE * BATCH_SIZE

print()
print("Total Number of TFRecord Files to Test :", test_set_size)
print("Total Number of Samples to Test:", test_samples)
print()

# Convert/Decode dataset from tfrecords file for training
test_objects = p4d.get_dataset(TEST_FILENAMES, shuffle=False, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=False)

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

def evaluate_predictions(objects):
    # Random Sample at every VAL_SAVE_FREQ
    JACCARD_LOSSES_all = []
    MSE_LOSSES_all = []
    JACCARD_LOSSES_all_NULL = []
    MSE_LOSSES_all_NULL = []
    JACCARD_LOSSES_all_INPUT = []
    MSE_LOSSES_all_INPUT = []

    for X in objects:
        
        X0 = tf.dtypes.cast(X[:,0], tf.float32)
        X1 = tf.dtypes.cast(X[:,1], tf.float32)

        # Encode & Generate
        latent_code = encoder(X0, training=False)
        pred_object = np.squeeze(generator(latent_code, training=False))

        # JACCARD FOR PREDICTIONS
        MSE_LOSSES_per_batch = MSE_reconstruction_loss(X1, pred_object).numpy()
        real = np.asarray(X1>0.5, bool)
        fake = np.asarray(pred_object>0.5, bool) 
        JACCARD_LOSSES_per_batch = JACCARD_reconstruction_loss(real, fake)
        
        # JACCARD AGAINST NULL MODEL (=MODEL THAT WILL PREDICT THE INPUT TIME FRAME EXACTLY)
        MSE_LOSSES_per_batch_NULL = MSE_reconstruction_loss(X1, X0).numpy()
        real = np.asarray(X1>0.5, bool)
        fake = np.asarray(X0>0.5, bool) 
        JACCARD_LOSSES_per_batch_NULL = JACCARD_reconstruction_loss(real, fake)
        
        # JACCARD AGAINST INPUT FRAME (=JACCARD BETWEEN PRED AND X0)
        MSE_LOSSES_per_batch_INPUT = MSE_reconstruction_loss(X0, pred_object).numpy()
        real = np.asarray(X0>0.5, bool)
        fake = np.asarray(pred_object>0.5, bool) 
        JACCARD_LOSSES_per_batch_INPUT = JACCARD_reconstruction_loss(real, fake)


        MSE_LOSSES_all.append(MSE_LOSSES_per_batch)
        JACCARD_LOSSES_all.append(JACCARD_LOSSES_per_batch)
        MSE_LOSSES_all_NULL.append(MSE_LOSSES_per_batch_NULL)
        JACCARD_LOSSES_all_NULL.append(JACCARD_LOSSES_per_batch_NULL)
        MSE_LOSSES_all_INPUT.append(MSE_LOSSES_per_batch_INPUT)
        JACCARD_LOSSES_all_INPUT.append(JACCARD_LOSSES_per_batch_INPUT)

        
    JACCARD_LOSSES_all_mean = np.mean(JACCARD_LOSSES_all)
    MSE_LOSSES_all_mean = np.mean(MSE_LOSSES_all)
    print("MEAN JACCARD LOSS OVER ALL SAMPLES: ", np.round(JACCARD_LOSSES_all_mean*100,3), "% | MSE ", np.round(MSE_LOSSES_all_mean,3))

    return np.array(MSE_LOSSES_all), np.array(MSE_LOSSES_all_NULL), np.array(MSE_LOSSES_all_INPUT), np.array(JACCARD_LOSSES_all), np.array(JACCARD_LOSSES_all_NULL), np.array(JACCARD_LOSSES_all_INPUT)


MSE_LOSSES_all, MSE_LOSSES_all_NULL, MSE_LOSSES_all_INPUT, JACCARD_LOSSES_all, JACCARD_LOSSES_all_NULL, JACCARD_LOSSES_all_INPUT = evaluate_predictions(test_objects)

MSE_LOSSES_all = np.reshape(MSE_LOSSES_all, (MSE_LOSSES_all.shape[0]*MSE_LOSSES_all.shape[1],1))
JACCARD_LOSSES_all = np.reshape(JACCARD_LOSSES_all, (JACCARD_LOSSES_all.shape[0]*JACCARD_LOSSES_all.shape[1],1))
MSE_LOSSES_all_NULL = np.reshape(MSE_LOSSES_all_NULL, (MSE_LOSSES_all_NULL.shape[0]*MSE_LOSSES_all_NULL.shape[1],1))
JACCARD_LOSSES_all_NULL = np.reshape(JACCARD_LOSSES_all_NULL, (JACCARD_LOSSES_all_NULL.shape[0]*JACCARD_LOSSES_all_NULL.shape[1],1))
MSE_LOSSES_all_INPUT= np.reshape(MSE_LOSSES_all_INPUT, (MSE_LOSSES_all_INPUT.shape[0]*MSE_LOSSES_all_INPUT.shape[1],1))
JACCARD_LOSSES_all_INPUT = np.reshape(JACCARD_LOSSES_all_INPUT, (JACCARD_LOSSES_all_INPUT.shape[0]*JACCARD_LOSSES_all_INPUT.shape[1],1))

LOSSES = np.concatenate((MSE_LOSSES_all, MSE_LOSSES_all_NULL, MSE_LOSSES_all_INPUT, JACCARD_LOSSES_all, JACCARD_LOSSES_all_NULL, JACCARD_LOSSES_all_INPUT), axis=1)

header = "          MSE_LOSS,        MSE_LOSS_NULL,       MSE_LOSS_INPUT,         JACCARD_LOSS,    JACCARD_LOSS_NULL,   JACCARD_LOSS_INPUT"
np.savetxt(f"evals/AEncGAN3D_TestSet_Errors_T{T}_TINC{TINC}.txt" , LOSSES, fmt="%20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f", delimiter=',', header=header)

