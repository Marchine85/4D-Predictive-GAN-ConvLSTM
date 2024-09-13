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
# OWN LIBRARIES
from utils import parse_tfrecords_4D_16_for_training_ConvLSTM as p4d


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

if not os.path.exists(os.path.join(base_dir,'evals')):
    os.makedirs(os.path.join(base_dir,'evals'))

MODEL_LOCATION      = os.path.join(base_dir, "convlstm3d_16_T80_TINC4")

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

print()
print("Total Number of TFRecord Files to Test :", test_set_size)
print()

# Convert/Decode dataset from tfrecords file for training
test_objects = p4d.get_dataset(TEST_FILENAMES, shuffle=False, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=False)

convlstm3d_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*ConvLSTM3D*.h5"))
convlstm3d = load_model(convlstm3d_model[-1])

def MSE_reconstruction_loss(real, fake):    
    squ = tf.square(real - fake)
    MSE_loss = tf.reduce_mean(squ, axis=[1,2,3,4])

    return MSE_loss

def JACCARD_reconstruction_loss(real, fake):    
    j_and = tf.math.reduce_sum(np.double(np.bitwise_and(real, fake)), axis=[1,2,3,4])
    j_or = tf.math.reduce_sum(np.double(np.bitwise_or(real, fake)), axis=[1,2,3,4])

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
    # Loop over tfrecords files / 4D Datapoints

    for X, y in objects:

        X = tf.dtypes.cast(X, tf.float32)
        y = tf.dtypes.cast(y, tf.float32)

        y_pred = np.squeeze(convlstm3d(X[...,np.newaxis], training=False))
        # JACCARD FOR PREDICTIONS
        MSE_LOSSES_per_batch = MSE_reconstruction_loss(y, y_pred).numpy()
        real = np.asarray(y>0.5, bool)
        fake = np.asarray(y_pred>0.5, bool) 
        JACCARD_LOSSES_per_batch = JACCARD_reconstruction_loss(real, fake)
        
        # JACCARD AGAINST NULL MODEL (=MODEL THAT WILL PREDICT THE INPUT TIME FRAME EXACTLY)
        MSE_LOSSES_per_batch_NULL = MSE_reconstruction_loss(X, y).numpy()
        real = np.asarray(X>0.5, bool)
        fake = np.asarray(y>0.5, bool) 
        JACCARD_LOSSES_per_batch_NULL = JACCARD_reconstruction_loss(real, fake)
        
        # JACCARD AGAINST INPUT FRAME (=JACCARD BETWEEN PRED AND X0)
        MSE_LOSSES_per_batch_INPUT = MSE_reconstruction_loss(X, y_pred).numpy()
        real = np.asarray(X>0.5, bool)
        fake = np.asarray(y_pred>0.5, bool) 
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
np.savetxt(f"evals/ConvLSTM3D_TestSet_Errors_T{T}_TINC{TINC}.txt" , LOSSES, fmt="%20.5f, %20.5f, %20.5f, %20.5f, %20.5f, %20.5f", delimiter=',', header=header)

