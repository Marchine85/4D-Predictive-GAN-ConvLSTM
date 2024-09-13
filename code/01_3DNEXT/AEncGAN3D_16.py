import os
from shutil import copyfile
import glob
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import time
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, LayerNormalization, LeakyReLU, Reshape, Conv3DTranspose, Conv3D, Flatten, ReLU, Dropout, Concatenate, Input, UpSampling3D, AveragePooling3D
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

# OWN LIBRARIES
from utils.loss_plotter import render_graphs
from utils import dataIO as d
from utils import parse_tfrecords_4D_16_for_training as p4d
from utils.logger import Logger

###############################################################################
########### INPUTS  ###################
#######################################
N_EPOCHS            = 10000             # Default: 10000 - Max. training epochs OR Define Max. Iterations
MAX_ITERATIONS      = 20000             # Default: -1 - Max. Training Iterations (Max. Number of processed Batches) before training stops, if "-1" Training stops after reaching N_EPOCHS
TRAIN_DATASET_LOC   = "Z:/Master_Thesis/data/00_4D_training_datasets"
VALID_DATASET_LOC   = "Z:/Master_Thesis/data/01_4D_validation_datasets"
N_TRAINING          = 10000             # Number of Files for Training: -1 == USE ALL
N_VALIDATION        = 1000              # Number of Files for Validation: -1 == USE ALL
T                   = 80                # Default: 80 - Even Number of Length of Time History to use, MUST BE EVENLY DIVISIBLE BY TINC, e.g. 80 means that 80 steps out of all 128 steps are used, where 24 steps at the beginning and 24 steps at the end of the time history are cropped.
TINC                = 4                 # Default: 1 - Allowed: (1, 2, 4, 8) - e.g.: 4 --> 80/4 = 20 Timepoints --> 19 PAIRS of T0->T1 --> TIME INCREMENT FOR EXTRACTING TIMESTEPS PER DATAPOINT -   (Check parse_tfrecords_4D_##_for_training for extraction details)
NORMALIZE           = False             # NORMALIZE DATA to ZERO MEAN --> from [0;1] to [-0.5, 0.5]
PRINT_D_LOSSES      = False             # LOG DISCRIMINATOR LOSSES SPLIT UP IN Fake, Real, GP and EPS Losses
LOGGING             = True              # LOG TRAINING RUN TO FILE (If Kernel get Interupted (user or error) Kernel needs to be restarted, otherwise log file is locked to be deleted!)
###############################################################################
# Restart Training?
RESTART_TRAINING    = False             
RESTART_ITERATION   = 50000             # the epoch number when the last training was stopped, has no influnce on loading the restart file, just used for tracking information numbering
RESTART_FOLDER_NAME = "20240221_012216_aencgan3d_16"
###############################################################################
# Output Settings
VAL_SAVE_FREQ       = 1000              # Default: 1000 - OFF: -1, every n training iterations (n batches) check validation loss
WRITE_RESTART_FREQ  = 1000              # Default: 1000 - every n training iterations (n batches) save restart checkpoint
KEEP_BEST_ONLY      = True              # Default: True - Keep only the best model (KEEP_LAST_N=1)
BEST_METRIC         = "JACCARD"         # Default: JACCARD - "JACCARD" or "MSE"
# if KEEP_BEST_ONLY = False define max. Number of last models to keep:
KEEP_LAST_N         = 5                 # Default: 5 - Max. number of last files to keep
N_SAMPLES           = 5                 # Default: 5 - Number of object samples to generate after each epoch
###############################################################################
########## HYPERPARAMETERS  ###########
#######################################
GD_ARCHITECTURE     = "PreRes"          # Default: "PreRes" - "PreRes" or "Classic" Convolutions
BATCH_SIZE          = 32                # Default: 32 - Number of Examples per Batch / Probably Increase for better Generalization
Z_SIZE              = 256               # Default: 256 - Latent Random Vector Size
FEATUREMAP_BASE_DIM = 256               # Default: 256 - Max Featuremap size (Maybe increase for learning larger/more diverse datasets...)
###############################################################################
# WGAN-GP Settings
GP_WEIGHT           = 100               # Default: 10 - WGAN-GP gradient penalty weight
EPS_WEIGHT          = 0.0               # Default: 0.001  - WGAN-GP epsilon penalty weight (real logits penalty)
D_STEPS             = 1                 # Default: 5 - WGAN-GP Disc. training steps
###############################################################################
# WGAN-GP Optimizer Settings
G_LR                = 0.0001            # Default: 0.0001 - Generator Learning Rate
D_LR                = 0.0001            # Default: 0.0001 - Discriminator Learning Rate
CLR                 = False             # Default: False - Use Cyclic Learning Rate, minLR=G_LR or D_LR, maxLR=10*G_LR or 10*D_LR
###############################################################################
# Dropout
DROPOUT_RATE        = 0.2               # Default: 0.2
###############################################################################
# weight_initializer & regularizer
weight_initializer  = "glorot_uniform"  # "glorot_uniform" "glorot_normal" or "he_uniform"
weight_regularizer  = "L1"              # tf.keras.regularizers.L1(l1=0.01)  # Default l1=0.01
###############################################################################
# USE BIAS IN GENERATOR
USE_BIAS            = True
bias_initializer    = "zeros" # DEFAULT: "zeros" - "ones" or tf.keras.initializers.Constant(0.5)
###############################################################################
# DEFAULT - DONT TOUCH SETTINGS
SIZE                = 16                # Default: 16 - Object Grid Size
BASE_SIZE           = 4                 # Default: 4 - Base Resolution to start from for generator and to end with for discriminator
KERNEL_SIZE         = 5                 # Default: 5 - Lower values learn pretty bad, higher than 5 doesnt change training much
###############################################################################
################### NETWORKS ##################################################
###############################################################################
# latent vector input
latent_vector = Input(shape=(Z_SIZE,))
#------------------------------------------------------------------------------
################### ENCODER ###################################################
#------------------------------------------------------------------------------
def obj_input(object_dim=(SIZE, SIZE, SIZE, 1)):
    obj_in = Input(shape=object_dim)
    return obj_in

###############################################################################
# Encoder PreResidual Block
def EncoderPreResBlock(x, in_filters, out_filters, kernel_size, strides, kernel_initializer, kernel_regularizer, padding):
    # Define Shortcut
    if strides == 1 and in_filters == out_filters: # Identity Skip Connection
        shortcut = x 
    elif strides == 2 and in_filters == out_filters: # Identity Skip Connection
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
    elif strides == 1 and in_filters != out_filters: # Change of Number of Filters Skip
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)
    elif strides == 2 and in_filters != out_filters: # Skip with Change in Size and Number of Filters
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding="same")(shortcut)

    # Define Activation on Input
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv3D(filters=out_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding=padding)(x)

    return shortcut + x
###############################################################################
# DEFINE ENCODER NETWORK
def ClassicEncoder():
    obj_in = obj_input()
    
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(obj_in)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)

    x = Flatten()(x)
    
    out_layer = Dense(Z_SIZE, activation=None)(x)

    # define model
    model = Model(obj_in, out_layer)
    return model

def PreResEncoder():
    obj_in = obj_input()

    # Begin Activation
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=1, input_shape=[SIZE, SIZE, SIZE, 1], kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(obj_in)

    # PreRes Blocks
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//8, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM   , kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Flatten (reshape) and Linear (dense) Layer
    x = Flatten()(x)

    out_layer = Dense(Z_SIZE, activation=None)(x)

    # define model
    model = Model(obj_in, out_layer)
    return model

###############################################################################

#------------------------------------------------------------------------------
################### GENERATOR #################################################
#------------------------------------------------------------------------------

def latent_input(latent_vector=latent_vector, BASE_SIZE=BASE_SIZE):
    base = BASE_SIZE * BASE_SIZE * BASE_SIZE

    latent_dense = Dense(units = FEATUREMAP_BASE_DIM*base)(latent_vector)
    latent_reshape = Reshape((BASE_SIZE, BASE_SIZE, BASE_SIZE, FEATUREMAP_BASE_DIM))(latent_dense)
    return latent_reshape

###############################################################################
# Generator PreResidual Block
def GPreResBlock(x, in_filters, out_filters, kernel_size, up_size, kernel_initializer, kernel_regularizer, padding, use_bias, bias_initializer):
    # Define Shortcut
    if up_size == 1 and in_filters == out_filters: # Identity Skip Connection
        shortcut = x 
    elif up_size == 2 and in_filters == out_filters: # Identity Skip Connection
        shortcut = UpSampling3D(size=(up_size, up_size, up_size))(x)
    elif up_size == 1 and in_filters != out_filters: # Change of Number of Filters Skip
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding=padding, use_bias=use_bias, bias_initializer = bias_initializer)(x)
    elif up_size == 2 and in_filters != out_filters: # Skip with Change in Size and Number of Filters
        shortcut = UpSampling3D(size=(up_size, up_size, up_size))(x)
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding=padding, use_bias=use_bias, bias_initializer = bias_initializer)(shortcut)

    # Define Activation on Input
    x = BatchNormalization()(x)
    x = ReLU()(x)     
    if up_size != 1:
        x = UpSampling3D(size=(up_size, up_size, up_size))(x)
    x = Conv3DTranspose(filters=out_filters, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding=padding, use_bias=use_bias, bias_initializer = bias_initializer)(x)

    return shortcut + x
###############################################################################
# DEFINE GENERATOR NETWORK
def ClassicGenerator():
    latent_in = latent_input()
    x = BatchNormalization()(latent_in)
    x = ReLU()(x)
    
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv3DTranspose(filters=1, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)(x)

    if NORMALIZE:
        out_layer = tanh(x)
    else:
        out_layer = sigmoid(x)

    model = Model(latent_vector, out_layer)
    
    return model

def PreResGenerator():
    x = latent_input()

    # PreRes Blocks
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM   , out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, up_size=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Final Convolution
    x = Conv3DTranspose(filters=1, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same", use_bias=USE_BIAS, bias_initializer = bias_initializer)(x)

    # End Activation
    if NORMALIZE:
        out_layer = tanh(x)
    else:
        out_layer = sigmoid(x)

    model = Model(latent_vector, out_layer)
    
    return model

###############################################################################

#------------------------------------------------------------------------------
################### DISCRIMINATOR #############################################
#------------------------------------------------------------------------------

def real_obj_input_disriminator(object_dim=(SIZE, SIZE, SIZE, 1)):
    real_obj_in = Input(shape=object_dim)
    return real_obj_in
def cond_obj_input_disriminator(object_dim=(SIZE, SIZE, SIZE, 1)):
    cond_obj_in = Input(shape=object_dim)
    return cond_obj_in

###############################################################################
# Discriminator PreResidual Block
def DPreResBlock(x, in_filters, out_filters, kernel_size, strides, kernel_initializer, kernel_regularizer, padding):
    # Define Shortcut
    if strides == 1 and in_filters == out_filters: # Identity Skip Connection
        shortcut = x 
    elif strides == 2 and in_filters == out_filters: # Identity Skip Connection
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
    elif strides == 1 and in_filters != out_filters: # Change of Number of Filters Skip
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)
    elif strides == 2 and in_filters != out_filters: # Skip with Change in Size and Number of Filters
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding="same")(shortcut)

    # Define Activation on Input
    x = LayerNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv3D(filters=out_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer, kernel_regularizer=weight_regularizer, padding=padding)(x)

    return shortcut + x
###############################################################################
# DEFINE DISCRIMINATOR NETWORK
def ClassicDiscriminator():
    real_obj_in = real_obj_input_disriminator()
    cond_obj_in = cond_obj_input_disriminator()
    
    # merge label_conditioned_generator and latent_input output
    x = Concatenate()([real_obj_in, cond_obj_in])

    x = Conv3D(filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)
    x = LayerNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)
    x = LayerNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)

    x = Flatten()(x)
    
    out_layer = Dense(1, activation=None)(x)

    # define model
    model = Model([real_obj_in, cond_obj_in], out_layer)
    return model

def PreResDiscriminator():
    real_obj_in = real_obj_input_disriminator()
    cond_obj_in = cond_obj_input_disriminator()
    
    # merge label_conditioned_disriminator and latent_input output
    x = Concatenate()([real_obj_in, cond_obj_in])

    # Begin Activation
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=1, input_shape=[SIZE, SIZE, SIZE, 1], kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")(x)

    # PreRes Blocks
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//8, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM   , kernel_size=KERNEL_SIZE, strides=1, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, padding="same")
    
    x = LayerNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Flatten (reshape) and Linear (dense) Layer
    x = Flatten()(x)

    out_layer = Dense(1, activation=None)(x)

    # define model
    model = Model([real_obj_in, cond_obj_in], out_layer)
    return model

###############################################################################
########## COMPILE NETWORKS ###################################################
###############################################################################
if GD_ARCHITECTURE == "PreRes":
    encoder, generator, discriminator = PreResEncoder(), PreResGenerator(), PreResDiscriminator()
elif GD_ARCHITECTURE == "Classic":
    encoder, generator, discriminator = ClassicEncoder(), ClassicGenerator(), ClassicDiscriminator()
else:
    ValueError("Generator/Disriminator Architecture not Available. Choose between 'PreRes' or 'Classic'!")
encoder.compile()  
generator.compile()  
discriminator.compile()  

###############################################################################
############## LOSS FUNCTION ##################################################
###############################################################################
@tf.function
def encgen_loss(real_objects):
    
    real_objects_t0 = tf.dtypes.cast(real_objects[:,0], tf.float32)
    real_objects_t1 = tf.dtypes.cast(real_objects[:,1], tf.float32)

    # Latent Encoding
    latent_code = encoder(real_objects_t0, training=True)

    pred_objects_t1 = generator(latent_code, training=True)
    fake_logits = discriminator([pred_objects_t1, real_objects_t1], training=True)
    loss = wgan_gp_generator_loss(fake_logits)
    
    mse = tf.keras.metrics.mean_squared_error(real_objects_t1, pred_objects_t1)
    MSE_loss = tf.reduce_mean(mse)
    
    return loss, MSE_loss

@tf.function
def encdisc_loss(real_objects):
    
    real_objects_t0 = tf.dtypes.cast(real_objects[:,0], tf.float32)
    real_objects_t1 = tf.dtypes.cast(real_objects[:,1], tf.float32)

    # Latent Encoding
    latent_code = encoder(real_objects_t0, training=True)

    pred_objects_t1 = generator(latent_code, training=True)
    fake_logits = discriminator([pred_objects_t1, real_objects_t1], training=True)
    real_logits = discriminator([real_objects_t1, real_objects_t1], training=True)
    D_loss, fake_loss, real_loss, gp, ep = wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects_t1, pred_objects_t1, real_objects_t1, GP_WEIGHT, EPS_WEIGHT)

    return D_loss, fake_loss, real_loss, gp, ep

@tf.function
def val_encdisc_loss(val_real_objects):
    
    val_real_objects_t0 = tf.dtypes.cast(val_real_objects[:,0], tf.float32)
    val_real_objects_t1 = tf.dtypes.cast(val_real_objects[:,1], tf.float32)

    # Latent Encoding
    val_latent_code = encoder(val_real_objects_t0, training=False)

    val_pred_objects_t1 = generator(val_latent_code, training=False)
    val_fake_logits = discriminator([val_pred_objects_t1, val_real_objects_t1], training=False)
    val_real_logits = discriminator([val_real_objects_t1, val_real_objects_t1], training=False)
    D_loss, fake_loss, real_loss, gp, ep = wgan_gp_discriminator_loss(val_real_logits, val_fake_logits, val_real_objects_t1, val_pred_objects_t1, val_real_objects_t1, GP_WEIGHT, EPS_WEIGHT)

    return D_loss, fake_loss, real_loss, gp, ep

# WGAN LOSSES
@tf.function
def wgan_gp_generator_loss(fake_logits):
    G_loss = -tf.math.reduce_mean(fake_logits)
    return G_loss
@tf.function
def wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects, fake_objects, cond_objects, GP_WEIGHT, EPS_WEIGHT):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # 1. Get the interpolated onject
    alpha = tf.random.uniform([real_objects.shape[0], 1, 1, 1, 1], minval=0.0, maxval=1.0, seed=None)
    differences = fake_objects - real_objects
    interpolates = real_objects + (alpha*differences)

    # 2. Calculate the gradients w.r.t to this interpolated object.
    gradients = tf.gradients(discriminator([interpolates, cond_objects], training=False), [interpolates])[0]
    norm = tf.math.sqrt(tf.math.reduce_sum(tf.square(gradients), axis=[1,2,3,4])) # norm over all gradients of 3D Grid

    # 3. Calculate the norm of the gradients.
    gradient_penalty = tf.reduce_mean((norm-1.0)**2) 

    # Add the gradient penalty to the original discriminator loss and MEAN over BATCH_SIZE + epsilon penalty
    fake_loss = tf.math.reduce_mean(fake_logits)
    real_loss = tf.math.reduce_mean(real_logits)
    gp = GP_WEIGHT * gradient_penalty
    ep = EPS_WEIGHT * tf.math.reduce_mean((real_logits)**2)
    
    D_loss =  fake_loss - real_loss + gp + ep

    return D_loss, fake_loss, real_loss, gp, ep

###############################################################################
############ TRAINING STEP ####################################################
###############################################################################
@tf.function
def train_step(obj_batch):

    real_objects = obj_batch[...,np.newaxis]
  
    for i in range(D_STEPS): 
        with tf.GradientTape() as disc_tape:
            # discriminator loss
            EncDisc_loss = encdisc_loss(real_objects)
            
        gradients_of_discriminator = disc_tape.gradient(EncDisc_loss[0], discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape:
        # generator loss
        EncGen_loss = encgen_loss(real_objects)
                
    gradients_of_encoder = enc_tape.gradient(EncGen_loss[0], encoder.trainable_variables)
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))

    gradients_of_generator = gen_tape.gradient(EncGen_loss[0], generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return EncGen_loss, EncDisc_loss

@tf.function
def validation_step(val_obj_batch):
    val_real_objects = val_obj_batch[...,np.newaxis]

    Val_EncDisc_Loss = val_encdisc_loss(val_real_objects)
    
    Val_MSE_reconstruction_loss = MSE_reconstruction_loss(val_real_objects)

    return Val_EncDisc_Loss, Val_MSE_reconstruction_loss


@tf.function
def MSE_reconstruction_loss(real_objects):    
    real_objects_t0 = tf.dtypes.cast(real_objects[:,0], tf.float32)
    real_objects_t1 = tf.dtypes.cast(real_objects[:,1], tf.float32)

    latent_code = encoder(real_objects_t0, training=False)
    pred_objects_t1 = generator(latent_code, training=False)

    mse = tf.keras.metrics.mean_squared_error(real_objects_t1, pred_objects_t1)
    MSE_loss = tf.reduce_mean(mse)

    return MSE_loss

def JACCARD_reconstruction_loss(val_obj_batch):    
    val_real_objects = val_obj_batch[...,np.newaxis]
    
    val_real_objects_t0 = tf.dtypes.cast(val_real_objects[:,0], tf.float32)
    val_real_objects_t1 = tf.dtypes.cast(val_real_objects[:,1], tf.float32)

    val_latent_code = encoder(val_real_objects_t0, training=False)
    val_pred_objects_t1 = generator(val_latent_code, training=False)

    if NORMALIZE:
        x = np.asarray(val_real_objects_t1>0.0, bool)
        y = np.asarray(val_pred_objects_t1>0.0, bool) 
    else:
        x = np.asarray(val_real_objects_t1>0.5, bool)
        y = np.asarray(val_pred_objects_t1>0.5, bool) 
            
    j_and = tf.math.reduce_sum(np.double(np.bitwise_and(x, y)), axis=[1,2,3,4])
    j_or = tf.math.reduce_sum(np.double(np.bitwise_or(x, y)), axis=[1,2,3,4])

    JACCARD_loss = tf.reduce_mean(np.nan_to_num(1.0 - j_and / j_or, nan=0.0))

    return JACCARD_loss

###############################################################################
######### SOME UTILITY FUNCTIONS ##############################################
###############################################################################
def generate_sample_objects(objects, total_iterations, tag="pred"):
    # Random Sample at every VAL_SAVE_FREQ
    i=0
    generated_objects = []
    for X in objects:
        X = tf.dtypes.cast(X, tf.float32)
        # Encode
        latent_code = encoder(X, training=False)

        sample_object = generator(latent_code, training=False)
        if NORMALIZE:
            object_out = np.squeeze(sample_object>0.0)
        else:
            object_out = np.squeeze(sample_object>0.5)
        
        try:
            d.plotMeshFromVoxels(object_out, obj=model_directory+tag+"_"+str(i)+"_next"+"_iter_"+str(total_iterations))
        except:
            print(f"Cannot generate STL, Marching Cubes Algo failed for sample {i}")
            # print("""Cannot generate STL, Marching Cubes Algo failed: Surface level must be within volume data range! \n
                    # This may happen at the beginning of the training, if it still happens at later stages epoch>10 --> Check Object and try to change Marching Cube Threshold.""")    
        
        generated_objects.append(object_out)
        
        i+=1
    
    print()
    
    return generated_objects

def IOhousekeeping(model_directory, KEEP_LAST_N, VAL_SAVE_FREQ, RESTART_TRAINING, total_iterations, KEEP_BEST_ONLY, best_iteration):
    if total_iterations > KEEP_LAST_N*VAL_SAVE_FREQ and total_iterations % VAL_SAVE_FREQ == 0:
        if KEEP_BEST_ONLY:
            fileList_all = glob.glob(model_directory + "*_iter_*")
            fileList_best = glob.glob(model_directory + "*_iter_" + str(int(best_iteration)) +"*")
            fileList_del = [ele for ele in fileList_all if ele not in fileList_best]
            
        else:
            fileList_del = glob.glob(model_directory + "*_iter_" + str(int(total_iterations-KEEP_LAST_N*VAL_SAVE_FREQ)) +"*")

        # Iterate over the list of filepaths & remove each file.
        for filePath in fileList_del:
            os.remove(filePath)

###############################################################################
########### LAST PREPARATION STEPS BEFORE TRAINING STARTS #####################
###############################################################################
#generate folders:
if RESTART_TRAINING:
    model_directory = os.getcwd() + "/" + RESTART_FOLDER_NAME + "/"
else:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    model_directory = os.getcwd()+"/"+time_string+"_" + os.path.splitext(os.path.basename(__file__))[0] + "/"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)  

# Save running python file to run directory
if RESTART_TRAINING:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    copyfile(__file__, model_directory+time_string +"_restart_"+os.path.basename(__file__))
else:
    copyfile(__file__, model_directory+os.path.basename(__file__))

# Activate Console Logging:
if LOGGING:
    sys.stdout = Logger(filename=model_directory+"log_out.txt")
        
# PRINT NETWORK SUMMARY
print(encoder.summary())
print(generator.summary())
print(discriminator.summary())

###############################################################################
############# GET DATA/OBJECTS ################################################
###############################################################################
TRAINING_FILENAMES = tf.io.gfile.glob(TRAIN_DATASET_LOC+"/*.tfrecords")[:N_TRAINING]
VALIDATION_FILENAMES = tf.io.gfile.glob(VALID_DATASET_LOC+"/*.tfrecords")[:N_VALIDATION]

training_set_size = len(TRAINING_FILENAMES)
validation_set_size = len(VALIDATION_FILENAMES)
training_samples = training_set_size * (T//TINC-1) // BATCH_SIZE * BATCH_SIZE
validation_samples = validation_set_size * (T//TINC-1) // BATCH_SIZE * BATCH_SIZE
print()
print("Total Number of TFRecord Files to Train :", training_set_size)
print("Total Number of Samples to Train:", training_samples)
print()
print("Total Number of TFRecord Files for Validation:", validation_set_size)
print("Total Number of Samples for Validation:", validation_samples)
print()

# Convert/Decode dataset from tfrecords file for training
train_objects = p4d.get_dataset(TRAINING_FILENAMES, shuffle=True, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=NORMALIZE )
val_objects = p4d.get_dataset(VALIDATION_FILENAMES, shuffle=False, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=NORMALIZE )

# save some real examples
i=0
test_objects = []
for X in train_objects.take(N_SAMPLES):
    Xt0 = X[0][0]
    Xt1 = X[0][1]

    try:
        sample_name = model_directory+"train_"+str(i)+"_init"#+str(t)
        d.plotMeshFromVoxels(np.array(Xt0), obj=sample_name)
        sample_name = model_directory+"train_"+str(i)+"_next"#+str(t)
        d.plotMeshFromVoxels(np.array(Xt1), obj=sample_name)
    except:
        print("Example Object couldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")

    test_objects.append([Xt0[...,np.newaxis]]) # for validation of reconstruction during training, see generate_sample_objects():
    i+=1    

# save some validation examples
i=0
valid_objects = []
for X in val_objects.take(N_SAMPLES):
    Xt0 = X[0][0]
    Xt1 = X[0][1]

    try:
        sample_name = model_directory+"valid_"+str(i)+"_init"#+str(t)
        d.plotMeshFromVoxels(np.array(Xt0), obj=sample_name)
        sample_name = model_directory+"valid_"+str(i)+"_next"#+str(t)
        d.plotMeshFromVoxels(np.array(Xt1), obj=sample_name)
    except:
        print("Example Object couldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")

    valid_objects.append([Xt0[...,np.newaxis]]) # for validation of reconstruction during training, see generate_sample_objects():
    i+=1    

###############################################################################
############## OPTIMIZER ######################################################
###############################################################################
if CLR:
    steps_per_epoch = len(TRAINING_FILENAMES) // BATCH_SIZE
    clr_g = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=G_LR,
        maximal_learning_rate=G_LR*10,
        scale_fn=lambda x: 1/(2.**(x-1)),
        step_size=2 * steps_per_epoch
    )
    clr_d = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=D_LR,
        maximal_learning_rate=D_LR*10,
        scale_fn=lambda x: 1/(2.**(x-1)),
        step_size=2 * steps_per_epoch
    )
    
    encoder_optimizer = Adam(learning_rate=clr_g, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
    generator_optimizer = Adam(learning_rate=clr_g, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
    discriminator_optimizer = Adam(learning_rate=clr_d, beta_1=0.5, beta_2=0.999, epsilon=1e-07)

else:
    encoder_optimizer = Adam(learning_rate=G_LR, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
    generator_optimizer = Adam(learning_rate=G_LR, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
    discriminator_optimizer = Adam(learning_rate=D_LR, beta_1=0.5, beta_2=0.999, epsilon=1e-07)

###############################################################################
# Define Checkpoint
checkpoint_path = model_directory+"checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt = tf.train.Checkpoint(encoder_optimizer=encoder_optimizer,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer,
                           encoder=encoder,
                           generator=generator,
                           discriminator=discriminator)

manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

def save_checkpoints(ckpt, model_directory):
    if not os.path.exists(model_directory+"checkpoints/"):
        os.makedirs(model_directory+"checkpoints/")  
    manager.save()

if RESTART_TRAINING:
    total_iterations = RESTART_ITERATION
    RESTART_EPOCH = training_samples // BATCH_SIZE
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("\nRestored from {} \n".format(manager.latest_checkpoint))
    else:
        print("No Checkpoint found --> Initializing from scratch.")

###############################################################################
############# TRAINING ########################################################
###############################################################################
print("----------------------------------------------------------------------")
print("RUN TRAINING:\n")
###############################################################################
start_time = time.time()
iter_arr, val_iter_arr  = [], []
G_losses = []
MSE_losses = []
JACCARD_losses = []
D_losses = []
D_val_loss, MSE_val_loss, JACCARD_val_loss = [], [], []     
n_objects = 0
total_iterations = 0
best_iteration = VAL_SAVE_FREQ
best_loss = 1e10
for epoch in range(1, N_EPOCHS+1):
    ###########################################################################
    # START OF EPOCH
    t_epoch_start = time.time()
    n_batch=0
    ##################################
    for obj_batch in train_objects:
        # START OF BATCH
        batch_time = time.time()   
        G_loss, D_loss = train_step(obj_batch)
        #Append arrays
        G_losses.append(G_loss[0])
        MSE_losses.append(G_loss[1])
        D_losses.append(-D_loss[0])
        
        n_batch+=1
        total_iterations+=1
        iter_arr.append(total_iterations)    

        JACCARD_loss = JACCARD_reconstruction_loss(obj_batch)
        JACCARD_losses.append(JACCARD_loss)
        print("E: %3d/%5d | B: %3d/%3d | I: %5d | T: %5.1f | dt: %2.2f || LOSSES: D %.4f | G %.4f | MSE %.4f | JACCARD %.4f" \
            % (epoch, N_EPOCHS, n_batch, np.ceil(training_set_size*(T//TINC-1)//BATCH_SIZE), total_iterations, time.time() - start_time, time.time() - batch_time,  -D_loss[0], G_loss[0], G_loss[1], JACCARD_loss))

        if PRINT_D_LOSSES:
            print("\nfake_loss: ", D_loss[1].numpy(), "real_loss: ", D_loss[2].numpy(), "gp: ", D_loss[3].numpy(), "ep: ", D_loss[4].numpy(), "\n\n")
                
        ###########################################################################
        # AFTER N TOTAL ITERATIONS GET VALIDATION LOSSES, SAVE GENERATED OBJECTS AND MODEL:
        if (total_iterations % VAL_SAVE_FREQ == 0) and VAL_SAVE_FREQ != -1:
            print("\nRUN VALIDATION, SAVE OBJECTS, MODEL AND PLOT LOSSES...")
                  
            Val_losses = []            
            Val_MSE_losses = []           
            Val_JACCARD_losses = []        
            for val_obj_batch in val_objects:
                Val_loss, MSE_recon = validation_step(val_obj_batch)
                Val_losses.append(-Val_loss[0]) # Collect Val loss per Batch
                Val_MSE_losses.append(MSE_recon) # Collect Val loss per Batch
                JACCARD_recon = JACCARD_reconstruction_loss(val_obj_batch)
                Val_JACCARD_losses.append(JACCARD_recon) # Collect Val loss per Batch
    
            D_val_loss.append(np.mean(Val_losses)) # Mean over batches
            MSE_val_loss.append(np.mean(Val_MSE_losses)) # Mean over batches
            JACCARD_val_loss.append(np.mean(Val_JACCARD_losses)) # Mean over batches
            val_iter_arr.append(total_iterations)
        ###########################################################################
            # Output generated objects
            generate_sample_objects(test_objects, total_iterations, "train")
            generate_sample_objects(valid_objects, total_iterations, "valid")
            # # Plot and Save Loss Curves
            render_graphs(model_directory, G_losses, D_losses, D_val_loss, MSE_losses, MSE_val_loss, JACCARD_losses, JACCARD_val_loss, iter_arr, val_iter_arr, RESTART_TRAINING) #this will only work after a 50 iterations to allow for proper averaging 

            if KEEP_BEST_ONLY:
                if BEST_METRIC == "JACCARD":
                    BEST_LOSS_METRIC = JACCARD_val_loss
                    KEEP_LAST_N = 1
                elif BEST_METRIC == "MSE":
                    BEST_LOSS_METRIC = MSE_val_loss
                    KEEP_LAST_N = 1
                if len(BEST_LOSS_METRIC) > 1 and (BEST_LOSS_METRIC[-1] <= best_loss):
                    best_loss = BEST_LOSS_METRIC[-1]
                    best_iteration = total_iterations
                    # Save models
                    encoder.save(model_directory+"_trained_encoder_"+"iter_"+str(total_iterations)+".h5")
                    generator.save(model_directory+"_trained_generator_"+"iter_"+str(total_iterations)+".h5")
            else:
                # Save models
                encoder.save(model_directory+"_trained_encoder_"+"iter_"+str(total_iterations)+".h5")
                generator.save(model_directory+"_trained_generator_"+"iter_"+str(total_iterations)+".h5")
                
            # # Delete Model, keep best or last N models
            IOhousekeeping(model_directory, KEEP_LAST_N, VAL_SAVE_FREQ, RESTART_TRAINING, total_iterations, KEEP_BEST_ONLY, best_iteration)  
            
        ###########################################################################
        # SAVE CHECKPOINT FOR RESTART:
        if total_iterations % WRITE_RESTART_FREQ == 0:
            print("WRITE RESTART FILE...\n")
            # Save Checkpoint after every 100 epoch
            save_checkpoints(ckpt, model_directory)
       
        # END OF BATCH
        ################################  
        # STOP TRAINING AFTER MAX DEFINED ITERATIONS ARE REACHED
        if total_iterations == MAX_ITERATIONS :
            break
    if total_iterations == MAX_ITERATIONS:
        break    
    
    # Print Status at the end of the epoch
    dt_epoch =  time.time() - t_epoch_start
    n_objects = n_objects + np.ceil(training_set_size*(T//TINC-1)//BATCH_SIZE)*BATCH_SIZE
    print("\n--------------------------------------------------------------------")
    print("END OF EPOCH ", epoch," | Total Training Iterations: ", int(total_iterations), " | Total Number of objects trained: ", int(n_objects/1000), "k", " | time elapsed:",  str(int((time.time() - start_time) / 60.0 )), "min")
    print("--------------------------------------------------------------------\n\n")

    # END OF EPOCH
    ###########################################################################

    
print("\n TRAINING DONE! \n") 
print("Total Training Time: ", str((time.time() - start_time) / 60.0), "min" )