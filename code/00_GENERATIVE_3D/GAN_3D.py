#!/usr/bin/env python
import os
from shutil import copyfile
import glob
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv3DTranspose, Conv3D, Flatten, ReLU, Dropout, Embedding, Concatenate, Input, UpSampling3D, AveragePooling3D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
# OWN LIBRARIES
from loss_plotter import render_graphs
import dataIO as d
import parse_tfrecords_3D_for_training as p3d

################################################################################################################################
#######################################
########### inputs  ###################
#######################################
N_EPOCHS            = 10000             # Default: 10000 - Max. epochs to run training
BATCH_SIZE          = 64                # Default: 64 (for 32x32x32 GridSize) - Number of Examples per Batch
# Latent Vector Size
Z_SIZE              = 32                # Default: 32 - Latent Random Vector Size (32 seems to be large enough,...)
# Objects Settings
SIZE                = 32                # Voxel Cube Size
BASE_SIZE           = 4                 # Base Resolution to start from for generator and to end with for discriminator
FEATUREMAP_BASE_DIM = 64                # Max Featuremap size (Maybe increase for learning larger/more diverse datasets...)
KERNEL_SIZE         = 5                 # Default: 5 -Lower values learn pretty bad, higher than 5 doesnt change training much
TRAIN_VAL_RATIO     = 0.005               # Fraction of objects of full dataset to be used for training (use 0.0005 to just draw 1 sample for training)
VALIDATION          = True              # Do Evaluation on Validation Set during Training
DATASET_LOCATION    = "D:/Master_Thesis/data/4D_128_32_topological_dataset_0-31_32"
N_LABELS            = 16 
###############################################################################
# Restart Training?
Restart_Training    = False
restart_epoch       = 1000
restart_folder_name = '20230415_110332_multiple_dataset_RasGANloss'
###############################################################################
# Output Settings
val_freq            = -1               # Default: 10 - OFF: -1, every n epochs check validation loss
save_freq           = 10                # Default: 10 - every n epochs print examples, save model, plot losses, do evalution of validation set
keep_last_N         = 10                # Default: 10 - Max. number of last files to keep
n_samples           = 10                # Default: 10 - Number of object samples to generate after each epoch
n_examples          = 10                # Default: 10 - Number of object examples to generate before training starts
###############################################################################
# Define Generator/Disriminator Architecture
GD_ARCHITECTURE     = "PreRes"          # Default: "PreRes" - "PreRes" or "Classic" Convolutions
###############################################################################
# Define LOSS    
LOSS                = 'WGAN-GP'         # Default: WGAN-GP - WGAN-GP / WGAN-GP-ADA / GAN / RasGAN 
###################
# WGAN-GP + WGAN-GP-Ada Only Settings
GP_WEIGHT           = 10                # Default: 10 - WGAN-GP gradient penalty weight
EPS_WEIGHT          = 0.001             # Default: 0.001  - WGAN-GP epsilon penalty weight (real logits penalty)
D_STEPS             = 1                 # Default: 5 - WGAN-GP Disc. training steps
# Additional WGAN-GP-ADA Settings
LAMBDA              = 1                 # Default: 1 - WGAN-GP-ADA - Ratio between Discriminator and Generator LOSS change rate for updating weights
###################
# GAN Only Settings
LABELNOISE         = True               # Default: True - Use LabelNoise within GAN LOSS Fucntion
NOISERATIO         = 0.02               # Default: 0.02 - Only if LABELNOISE = True; percentage of flip labels to add noise to gan LOSS
SMOOTHRATIO        = 0.1                # Default: 0.1  - label smoothing, to generate labels between [0 -- SMOOTHRATIO] and  [1-SMOOTHRATIO -- 1] 
###############################################################################
# WGAN-GP Optimizer Settings
G_LR               = 0.0002            # Default: 0.0002 (WGAN-GP & WGAN-GP-ADA) - Generator Learning Rate
D_LR               = 0.0002            # Default: 0.0002 (WGAN-GP & WGAN-GP-ADA) - Discriminator Learning Rate
#
# Reccommended GAN optimization parameters
# G_LR       = 0.001                    # Generator Learning Rate
# D_LR       = 0.001                    # Discriminator Learning Rate
# Reccommended RasGAN optimization parameters
# G_LR       = 0.002                    # Generator Learning Rate
# D_LR       = 0.00005                  # Discriminator Learning Rate
###############################################################################
# Dropout
DROPOUT             = True              # Default: True - Use Dropout inside Discriminator
# if dropout True, define Dropout Rate
DROPOUT_RATE        = 0.2               # Default: 0.2
###############################################################################
# Global Seed for Randomization
tf.random.set_seed(None)
###############################################################################
# weight_initializer
weight_initializer = 'glorot_uniform'   # "glorot_uniform" or "he_uniform"
###################
# Set BatchNorm depending on loss function
if (LOSS != "WGAN-GP") and (LOSS != "WGAN-GP-ADA"):
    BATCHNORM=True
else:
    BATCHNORM=False
####################################################################################################################
################### NETWORKS ##################################################
###############################################################################
# label input
con_label = Input(shape=(1,))
# latent vector input
latent_vector = Input(shape=(Z_SIZE,))
###############################################################################
# DEFINE Conditional Layers for Generator
def label_conditioned_generator(N_LABELS=N_LABELS, embedding_dim=10):
    # embedding for categorical input
    label_embedding = Embedding(N_LABELS, embedding_dim)(con_label)
    # linear multiplication
    base = BASE_SIZE * BASE_SIZE * BASE_SIZE
    label_dense = Dense(base)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = Reshape((BASE_SIZE, BASE_SIZE, BASE_SIZE, 1))(label_dense)
    return label_reshape_layer

def latent_input(latent_vector=latent_vector, BASE_SIZE=BASE_SIZE, FEATUREMAP_BASE_DIM=FEATUREMAP_BASE_DIM):
    base = BASE_SIZE * BASE_SIZE * BASE_SIZE
    latent_dense = Dense(units = FEATUREMAP_BASE_DIM*base)(latent_vector)
    latent_reshape = Reshape((BASE_SIZE, BASE_SIZE, BASE_SIZE, FEATUREMAP_BASE_DIM))(latent_dense)
    return latent_reshape
###############################################################################
# Generator PreResidual Block
def GPreResBlock(x, in_filters, out_filters, kernel_size, up_size, kernel_initializer, padding, use_bias):
    # Define Shortcut
    if up_size == 1 and in_filters == out_filters: # Identity Skip Connection
        shortcut = x 
    elif up_size == 2 and in_filters == out_filters: # Identity Skip Connection
        shortcut = UpSampling3D(size=(up_size, up_size, up_size))(x)
    elif up_size == 1 and in_filters != out_filters: # Change of Number of Filters Skip
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    elif up_size == 2 and in_filters != out_filters: # Skip with Change in Size and Number of Filters
        shortcut = UpSampling3D(size=(up_size, up_size, up_size))(x)
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(shortcut)

    # Define Activation on Input
    x = UpSampling3D(size=(up_size, up_size, up_size))(x)
    x = Conv3DTranspose(filters=out_filters, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)   
    x = Conv3DTranspose(filters=out_filters, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)   

    return shortcut + x
###############################################################################
# DEFINE GENERATOR NETWORK
def ClassicGenerator():
    label_output = label_conditioned_generator()
    latent_vector_output = latent_input()
    
    # merge label_conditioned_generator and latent_input output
    x = Concatenate()([latent_vector_output, label_output])
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv3DTranspose(filters=1, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same', use_bias=False)(x)

    out_layer = sigmoid(x)

    model = Model([latent_vector, con_label], out_layer)
    
    return model

def PreResGenerator():
    label_output = label_conditioned_generator()
    latent_vector_output = latent_input()
    
    # merge label_conditioned_generator and latent_input output
    x = Concatenate()([latent_vector_output, label_output])
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # PreRes Blocks
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM,    out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)
    x = GPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)

    # Final Convolution
    x = Conv3DTranspose(filters=1, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same')(x)

    # End Activation
    out_layer = sigmoid(x)

    model = Model([latent_vector, con_label], out_layer)
    
    return model

###############################################################################
# DEFINE Conditional Layers for Discriminator
def label_conditioned_discriminator(object_dim=(SIZE, SIZE, SIZE, 1), N_LABELS=N_LABELS, embedding_dim=10):
   
    # embedding for categorical input
    label_embedding = Embedding(N_LABELS, embedding_dim)(con_label)
    #print(label_embedding)
    # linear multiplication
    nodes = SIZE*SIZE*SIZE
    label_dense = Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = Reshape((SIZE, SIZE, SIZE, 1))(label_dense)
    return label_reshape_layer

def obj_input_disriminator(object_dim=(SIZE, SIZE, SIZE, 1)):
    obj_in = Input(shape=object_dim)
    return obj_in
###############################################################################
# Discriminator PreResidual Block
def DPreResBlock(x, in_filters, out_filters, kernel_size, strides, kernel_initializer, padding):
    # Define Shortcut
    if strides == 1 and in_filters == out_filters: # Identity Skip Connection
        shortcut = x 
    elif strides == 2 and in_filters == out_filters: # Identity Skip Connection
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
    elif strides == 1 and in_filters != out_filters: # Change of Number of Filters Skip
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, padding='same')(x)
    elif strides == 2 and in_filters != out_filters: # Skip with Change in Size and Number of Filters
        shortcut = AveragePooling3D(pool_size=(strides, strides, strides), strides=(strides, strides, strides))(x)
        shortcut = Conv3D(filters=out_filters, kernel_size=1, strides=1, kernel_initializer=kernel_initializer, padding='same')(shortcut)

    # Define Activation on Input
    x = Conv3D(filters=out_filters, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding)(x)

    if BATCHNORM:
        x = BatchNormalization()(x)
        
    x = LeakyReLU()(x)
    
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=out_filters, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer, padding=padding)(x)

    if BATCHNORM:
        x = BatchNormalization()(x)
        
    x = LeakyReLU()(x)
    
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)

    return shortcut + x
###############################################################################
# DEFINE DISCRIMINATOR NETWORK
def ClassicDiscriminator():
    label_conditioned_discriminator_in = label_conditioned_discriminator()
    obj_in = obj_input_disriminator()
    
    # merge label_conditioned_generator and latent_input output
    x = Concatenate()([obj_in, label_conditioned_discriminator_in])

    x = Conv3D(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=2, input_shape=[SIZE, SIZE, SIZE, 1], kernel_initializer = weight_initializer, padding='same')(x)
    if BATCHNORM:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)
    
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')(x)
    if BATCHNORM:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')(x)
    if BATCHNORM:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)

    x = Conv3D(filters=FEATUREMAP_BASE_DIM, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same')(x)

    x = Flatten()(x)
    
    out_layer = Dense(1, activation=None)(x)

    # define model
    model = Model([obj_in, con_label], out_layer)
    return model

def PreResDiscriminator():
    label_conditioned_discriminator_in = label_conditioned_discriminator()
    obj_in = obj_input_disriminator()
    
    # merge label_conditioned_disriminator and latent_input output
    x = Concatenate()([obj_in, label_conditioned_discriminator_in])

    # Begin Activation
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=2, input_shape=[SIZE, SIZE, SIZE, 1], kernel_initializer = weight_initializer, padding='same')(x)
    if BATCHNORM:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)
    
    # PreRes Blocks
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//8, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')
    x = DPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM,    kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same')
    
    # Flatten (reshape) and Linear (dense) Layer
    x = Flatten()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)
    out_layer = Dense(1, activation=None)(x)

    # define model
    model = Model([obj_in, con_label], out_layer)
    return model
#############################################################################################################################################################
###############################################################################
########## COMPILE NETWORKS ###################################################
###############################################################################
if GD_ARCHITECTURE == "PreRes":
    generator, discriminator = PreResGenerator(), PreResDiscriminator()
elif GD_ARCHITECTURE == "Classic":
    generator, discriminator = ClassicGenerator(), ClassicDiscriminator()
else:
    ValueError("Generator/Disriminator Architecture not Available. Choose between 'PreRes' or 'Classic'!")
generator.compile()  
print(generator.summary())
discriminator.compile()  
print(discriminator.summary())

##############################################################################################################################################################
###############################################################################
############## Optimization ###################################################
###############################################################################
if LOSS == "WGAN-GP":
    generator_optimizer = Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.9, epsilon=1e-07)
    discriminator_optimizer = Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.9, epsilon=1e-07)

elif LOSS == "WGAN-GP-ADA":
    generator_optimizer = Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.9, epsilon=1e-07)
    discriminator_optimizer = Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.9, epsilon=1e-07)

else:
    generator_optimizer = Adam(learning_rate=G_LR, beta_1=0.5, beta_2=0.99, epsilon=1e-07)
    discriminator_optimizer = Adam(learning_rate=D_LR, beta_1=0.5, beta_2=0.99, epsilon=1e-07)
    
g_vars = generator.trainable_variables
d_vars = discriminator.trainable_variables

print('\nG-vars count= ' + str(len(g_vars)))
print('D-vars count= ' + str(len(d_vars)))

##############################################################################################################################################################
###############################################################################
############## LOSS Function ##################################################
###############################################################################
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=SMOOTHRATIO)
#------------------------------------------------------------------------------
# LEGACY LOSSES
# ### Randomly flip labels to introduce more noise to the discriminator
# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)
    # invert the labels in place
    op_list = []
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1.0, y[i]))
        else:
            op_list.append(y[i])
    
    outputs = tf.stack(op_list)
    return outputs

def generator_loss(real_logits, fake_logits, LOSS='GAN'):
    if LOSS == 'GAN':
        G_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)

    elif LOSS == 'RasGAN':            
        real_tilde = sigmoid(real_logits - tf.math.reduce_mean(fake_logits))
        fake_tilde = sigmoid(fake_logits - tf.math.reduce_mean(real_logits))
        # RASGAN LOSSES:
        G_loss = - tf.math.reduce_mean(tf.math.log(fake_tilde + 1e-14)) - tf.math.reduce_mean(tf.math.log(1 - real_tilde + 1e-14))
        
    return G_loss

def discriminator_loss(real_logits, fake_logits, LOSS='GAN'):
    if LOSS == 'GAN':        #add noise to labels
        if LABELNOISE == True:
            real_logits_labels = noisy_labels(tf.ones_like(real_logits), NOISERATIO)
            fake_logits_labels = noisy_labels(tf.zeros_like(fake_logits), NOISERATIO)
        else:
            real_logits_labels = tf.ones_like(real_logits)
            fake_logits_labels = tf.zeros_like(fake_logits)
        #LOSS
        real_loss = cross_entropy(real_logits_labels, real_logits)
        fake_loss = cross_entropy(fake_logits_labels, fake_logits)
        D_loss = real_loss + fake_loss

    elif LOSS == 'RasGAN':
        real_tilde = sigmoid(real_logits - tf.math.reduce_mean(fake_logits))
        fake_tilde = sigmoid(fake_logits - tf.math.reduce_mean(real_logits))
        D_loss = - tf.math.reduce_mean(tf.math.log(real_tilde + 1e-14)) - tf.math.reduce_mean(tf.math.log(1 - fake_tilde + 1e-14))
        
    return D_loss    

#------------------------------------------------------------------------------
# WGAN LOSSES
def wgan_gp_generator_loss(fake_logits):
    G_loss = -tf.math.reduce_mean(fake_logits)
    return G_loss

def wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects, fake_objects, real_labels, GP_WEIGHT, EPS_WEIGHT):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    real_objects = tf.dtypes.cast(real_objects, tf.float32)
    # 1. Get the interpolated image
    alpha = tf.random.uniform([real_objects.shape[0], 1, 1, 1, 1], minval=0.0, maxval=1.0, seed=None)
    differences = fake_objects - real_objects
    interpolates = real_objects + (alpha*differences)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    gradients = tf.gradients(discriminator([interpolates, real_labels]), [interpolates])[0]
    norm = tf.math.sqrt(tf.math.reduce_sum(tf.square(gradients), axis=[1,2,3,4])) # norm over all gradients of 3D Grid

    # 3. Calculate the norm of the gradients.
    gradient_penalty = tf.reduce_mean((norm-1.0)**2) 

    # Add the gradient penalty to the original discriminator loss and MEAN over BATCH_SIZE + epsilon penalty
    D_loss =  tf.math.reduce_mean(fake_logits) - tf.math.reduce_mean(real_logits) + GP_WEIGHT * gradient_penalty + EPS_WEIGHT * tf.math.reduce_mean((real_logits)**2)

    return D_loss    

print('LOSS FUNCTION: ', LOSS + '\n')

####################################################################################################################
####################################################################################################################
###############################################################################
# SOME UTILITY FUNCTIONS
def generate_sample_objects(n_samples, N_LABELS, Z_SIZE, epoch):
    # Random Sample at every save_freq
    z_samples = tf.random.normal([n_samples, Z_SIZE], mean=0.0, stddev=1.0)
    label_samples = np.random.randint(1, N_LABELS+1, size=(n_samples,1))
    
    sample_objects = generator([z_samples, label_samples], training=False)
    for i in range(n_samples):
        objects_out = np.squeeze(sample_objects[i]>0.5)
        try:
            #np.save(model_directory + 'np_' + str(epoch) +'_'+str(i), objects_out)
            d.plotMeshFromVoxels(objects_out, obj=model_directory+'sample_epoch_'+str(epoch)+'_label_'+str(label_samples[i][0])+'_'+str(i+1))
        except:
            print('''Cannot generate STL, Marching Cubes Algo failed: Surface level must be within volume data range! \n
                    This may happen at the beginning of the training, if it still happens at later stages epoch>10 --> Check Object and try to change Marching Cube Threshold.''')    

def IOhousekeeping(model_directory, keep_last_N, save_freq, Restart_Training, epoch):
    if epoch > keep_last_N*save_freq and epoch % save_freq == 0:
        fileList = glob.glob(model_directory + '*_epoch_' + str(int(epoch-keep_last_N*save_freq)) +'*')
        # Iterate over the list of filepaths & remove each file.
        for filePath in fileList:
            os.remove(filePath)
            if not Restart_Training: # Only delete in Main Run, when Restart, keep the old ones, to see the loss history from the main run
                if os.path.exists(model_directory + 'plots/' + str(int(epoch-keep_last_N*save_freq)) +'.png'):    
                    os.remove(model_directory + 'plots/' + str(int(epoch-keep_last_N*save_freq)) +'.png')

####################################################################################################################
###############################################################################
# TRAINING STEP 
###############################################################################
# INIT None variables to further allow update inside tf.function decorator
if LOSS=='WGAN-GP-ADA':
    G_loss_WGAN_GP_ada_prev = None
    D_loss_WGAN_GP_ada_prev = None
    G_loss_WGAN_GP_ada_curr = None
    D_loss_WGAN_GP_ada_curr = None
    rg = None
    rd = None
    
elif LOSS=='RasGAN':
    d_train_counter = None
    D_loss_RasGAN_curr = None
    D_loss_RasGAN_prev = None
    G_loss_smooth = None
    D_loss_smooth = None
    alpha = None

@tf.function
def train_step(obj_batch, label_batch, LOSS):
    if LOSS=='GAN':
        real_objects = obj_batch[...,np.newaxis]
        real_labels = label_batch[...,np.newaxis]

        # Define random Latent Vec z to generate fake/training objects
        z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_objects = generator([z, real_labels], training=True)
            fake_logits = discriminator([fake_objects, real_labels], training=True)
            real_logits = discriminator([real_objects, real_labels], training=True)

            #######################################
            #discriminator LOSS
            D_loss_GAN = discriminator_loss(real_logits, fake_logits, LOSS=LOSS)
            #######################################
            #generator LOSS
            G_loss_GAN = generator_loss(real_logits, fake_logits, LOSS=LOSS)

        gradients_of_generator = gen_tape.gradient(G_loss_GAN, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(D_loss_GAN, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return G_loss_GAN, D_loss_GAN
    

    elif LOSS=='RasGAN':
        # Initialize tf-Variables to TF Graph to work with @tf.function decorator  
        global d_train_counter, D_loss_RasGAN_curr, D_loss_RasGAN_prev, G_loss_smooth, D_loss_smooth, alpha
        if d_train_counter is None:
            tf.print("INIT")
            d_train_counter = tf.Variable(0)
            D_loss_RasGAN_curr = tf.Variable(0.0)
            D_loss_RasGAN_prev = tf.Variable(0.0)
            G_loss_smooth = tf.Variable(0.0)
            D_loss_smooth = tf.Variable(1.0)
            alpha = tf.Variable(0.3) 
            
        real_objects = obj_batch[...,np.newaxis]
        real_labels = label_batch[...,np.newaxis]

        # Define random Latent Vec z to generate fake/training objects
        z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_objects = generator([z, real_labels], training=True)
            fake_logits = discriminator([fake_objects, real_labels], training=True)
            real_logits = discriminator([real_objects, real_labels], training=True)
            #training the discriminator 
            if (2 * D_loss_smooth) > G_loss_smooth: 
                # #discriminator logits
                # real_logits = discriminator([real_objects, real_labels], training=True)
                #discriminator loss
                D_loss_RasGAN = discriminator_loss(real_logits, fake_logits, LOSS=LOSS) 
                d_train_counter.assign(0)
            else:
                D_loss_RasGAN_curr.assign(D_loss_RasGAN_prev)
                D_loss_RasGAN = D_loss_RasGAN_prev
  
            #generator loss
            G_loss_RasGAN = generator_loss(real_logits, fake_logits, LOSS=LOSS) 
            
        gradients_of_generator = gen_tape.gradient(G_loss_RasGAN, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        #training the discriminator 
        if (2 * D_loss_smooth) > G_loss_smooth:             
            gradients_of_discriminator = disc_tape.gradient(D_loss_RasGAN, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
                    
		# Train discriminator on the next round?
        G_loss_smooth.assign( (alpha * G_loss_RasGAN) + ((1.0 - alpha) * G_loss_smooth) )
        D_loss_smooth.assign( (alpha * D_loss_RasGAN) + ((1.0 - alpha) * D_loss_smooth) )
        d_train_counter.assign(d_train_counter+1)

        D_loss_RasGAN_prev.assign(D_loss_RasGAN)

        return G_loss_RasGAN, D_loss_RasGAN
    

    elif LOSS=='WGAN-GP':
        real_objects = obj_batch[...,np.newaxis]
        real_labels = label_batch[...,np.newaxis]
  
        for i in range(D_STEPS):
            # Define random Latent Vec z to generate fake/training objects
            z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)
 
            with tf.GradientTape() as disc_tape:
                fake_objects = generator([z, real_labels], training=True)
                fake_logits = discriminator([fake_objects, real_labels], training=True)
                real_logits = discriminator([real_objects, real_labels], training=True)
                #discriminator loss
                D_loss_WGAN_GP = wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects, fake_objects, real_labels, GP_WEIGHT, EPS_WEIGHT)
                
            gradients_of_discriminator = disc_tape.gradient(D_loss_WGAN_GP, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
        # Define random Latent Vec z to generate fake/training objects
        z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)
  
        with tf.GradientTape() as gen_tape:
            fake_objects = generator([z, real_labels], training=True)
            fake_logits = discriminator([fake_objects, real_labels], training=True)
            #generator loss
            G_loss_WGAN_GP = wgan_gp_generator_loss(fake_logits)
        
        gradients_of_generator = gen_tape.gradient(G_loss_WGAN_GP, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        
        return G_loss_WGAN_GP, -D_loss_WGAN_GP


    elif LOSS=='WGAN-GP-ADA':
        real_objects = obj_batch[...,np.newaxis]
        real_labels = label_batch[...,np.newaxis]
  
        # Define random Latent Vec z to generate fake/training objects
        z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)
 
        with tf.GradientTape() as disc_tape:
            fake_objects = generator([z, real_labels], training=True)
            fake_logits = discriminator([fake_objects, real_labels], training=True)
            real_logits = discriminator([real_objects, real_labels], training=True)
            #discriminator loss
            D_loss_WGAN_GP_ada = wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects, fake_objects, real_labels, GP_WEIGHT, EPS_WEIGHT)

        # Define random Latent Vec z to generate fake/training objects
        z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)
  
        with tf.GradientTape() as gen_tape:
            fake_objects = generator([z, real_labels], training=True)
            fake_logits = discriminator([fake_objects, real_labels], training=True)
            #generator loss
            G_loss_WGAN_GP_ada = wgan_gp_generator_loss(fake_logits)
    
        # Initialize tf-Variables to TF Graph to work with @tf.function decorator        
        global G_loss_WGAN_GP_ada_prev, D_loss_WGAN_GP_ada_prev, G_loss_WGAN_GP_ada_curr, D_loss_WGAN_GP_ada_curr, rg, rd
        if rg is None:
            tf.print("INIT")
            G_loss_WGAN_GP_ada_prev = tf.Variable(G_loss_WGAN_GP_ada)
            D_loss_WGAN_GP_ada_prev = tf.Variable(D_loss_WGAN_GP_ada)
            G_loss_WGAN_GP_ada_curr = tf.Variable(G_loss_WGAN_GP_ada)
            D_loss_WGAN_GP_ada_curr = tf.Variable(D_loss_WGAN_GP_ada)
            rg = tf.Variable(1.0)
            rd = tf.Variable(1.0)

        if rd > LAMBDA * rg: # update discriminator     
            #print("Update Discriminator")           
            gradients_of_discriminator = disc_tape.gradient(D_loss_WGAN_GP_ada, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        else: # update generator   
            #print("Update Generator") 
            gradients_of_generator = gen_tape.gradient(G_loss_WGAN_GP_ada, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        G_loss_WGAN_GP_ada_curr.assign( G_loss_WGAN_GP_ada)
        D_loss_WGAN_GP_ada_curr.assign( D_loss_WGAN_GP_ada)
            
        rg.assign( tf.abs((G_loss_WGAN_GP_ada_curr - G_loss_WGAN_GP_ada_prev) / G_loss_WGAN_GP_ada_prev) )
        rd.assign( tf.abs((D_loss_WGAN_GP_ada_curr - D_loss_WGAN_GP_ada_prev) / D_loss_WGAN_GP_ada_prev) )
        
        #Prev Loss for next step rate calculation
        G_loss_WGAN_GP_ada_prev.assign( G_loss_WGAN_GP_ada)
        D_loss_WGAN_GP_ada_prev.assign( D_loss_WGAN_GP_ada)
            
        return G_loss_WGAN_GP_ada, -D_loss_WGAN_GP_ada

@tf.function
def validation_step(val_obj_batch, val_label_batch, LOSS):
    real_objects = val_obj_batch[...,np.newaxis]
    real_labels = val_label_batch[...,np.newaxis]

    # Define random Latent Vec z to generate fake/training objects
    z = tf.random.normal([real_objects.shape[0], Z_SIZE], mean=0.0, stddev=1.0)

    fake_objects = generator([z, real_labels], training=False)
    fake_logits = discriminator([fake_objects, real_labels], training=False)
    real_logits = discriminator([real_objects, real_labels], training=False)
    
    if (LOSS=='GAN') or (LOSS=='RasGAN'):
        return discriminator_loss(real_logits, fake_logits, LOSS=LOSS)
    
    elif (LOSS=='WGAN-GP') or ('WGAN-GP-ADA'):       
        return -wgan_gp_discriminator_loss(real_logits, fake_logits, real_objects, fake_objects, real_labels, GP_WEIGHT, EPS_WEIGHT)

####################################################################################################################
##########################################################################
#####################################
# LAST PREPARATION STEPS BEFORE TRAINING STARTS
##########################################################
#####################################
#generate folders:
if Restart_Training:
    model_directory = os.getcwd() + '/' + restart_folder_name + '/'
else:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    model_directory = os.getcwd()+'/'+time_string+'_' + LOSS + '-loss' + '/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)  

# Save running python file to run directory
if Restart_Training:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    copyfile(__file__, model_directory+time_string +'_restart_'+os.path.basename(__file__))
else:
    copyfile(__file__, model_directory+os.path.basename(__file__))

###############################################################################
############# GET Data/Objects ################################################
###############################################################################

FILENAMES = tf.io.gfile.glob(DATASET_LOCATION+"/*.tfrecords")[::10]
split_ind = int(TRAIN_VAL_RATIO * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]
training_set_size = len(TRAINING_FILENAMES)
validation_set_size = len(VALID_FILENAMES)
print()
print("Train TFRecord Files:", training_set_size)
print("Validation TFRecord Files:", validation_set_size)

# Convert/Decode dataset from tfrecords file for training
objects = p3d.get_dataset(TRAINING_FILENAMES, shuffle=True, batch_size=BATCH_SIZE )
val_objects = p3d.get_dataset(VALID_FILENAMES, shuffle=False, batch_size=BATCH_SIZE )


# save some real examples
i=0
for X, y in objects.take(n_examples):
    #print('X:', X, 'y:', y.numpy())
    try:
        sample_name = model_directory+'label_'+str(y.numpy()[0])+'_'+str(i)+'ex'
        d.plotMeshFromVoxels(np.array(X[0]), obj=sample_name)
    except:
        print("Example Object youldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")
    i+=1

if Restart_Training:
    restart_epoch = restart_epoch
else:
    restart_epoch = 0

###############################################################################
# Define Checkpoint
checkpoint_path = model_directory+'checkpoints/'
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

def save_checkpoints(ckpt, model_directory, epoch):
    if not os.path.exists(model_directory+'checkpoints/'):
        os.makedirs(model_directory+'checkpoints/')  
    if epoch % 10 == 0:   
        ckpt.save(model_directory+'checkpoints/ckpt')

if Restart_Training:
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("\nRestored from {} \n".format(manager.latest_checkpoint))
    else:
        print("No Checkpoint found --> Initializing from scratch.")

####################################################################################################################
##########################################################################
############# Training ##############
#####################################
print()
print('RUN TRAINING:\n')
###############################################################################
start_time = time.time()
track_g_loss, track_d_loss, track_d_val_loss, epoch_arr, val_epoch_arr = [], [], [], [], []
n_objects = 0
for epoch in range(1+restart_epoch, 1+N_EPOCHS):
    ###########################################################################
    # START OF EPOCH
    t_epoch_start = time.time()
    n_batch=1
    G_losses = []
    D_losses = []
    for obj_batch, label_batch in objects:
        ##################################
        # START OF BATCH
        batch_time = time.time()   
        G_loss, D_loss = train_step(obj_batch, label_batch, LOSS)
        #Append arrays
        G_losses.append(G_loss)
        D_losses.append(D_loss)
        print("Epoch: [%2d/%2d] Batch: [%4d/%4d] elapsed_time: %.1f, dt_batch: %.2f, d_loss: %.4f, g_loss: %.4f" \
            % (epoch, N_EPOCHS, n_batch, training_set_size//BATCH_SIZE, time.time() - start_time, time.time() - batch_time,  D_loss, G_loss))
        n_batch+=1
        # END OF BATCH
        ################################
    track_g_loss.append(np.mean(G_losses))  
    track_d_loss.append(np.mean(D_losses))   
    epoch_arr.append(epoch)        
    
    ###########################################################################
    # Get Validation Loss
    if (epoch % val_freq == 0) and val_freq != -1:
        Val_losses = []            
        for val_obj_batch, val_label_batch in val_objects:
            Val_loss = validation_step(val_obj_batch, val_label_batch, LOSS)
            Val_losses.append(Val_loss) # Collect Val loss per Batch
        track_d_val_loss.append(np.mean(Val_losses)) # Mean over batches
        val_epoch_arr.append(epoch)
    ###########################################################################
    ###########################################################################
    # AFTER EACH EPOCH TRAINING IS DONE:
        
    if epoch % save_freq == 0:
        # Output generated objects
        generate_sample_objects(n_samples, N_LABELS, Z_SIZE, epoch)
        # # Plot and Save Loss Curves
        render_graphs(model_directory, epoch, track_g_loss, track_d_loss, track_d_val_loss, epoch_arr, val_epoch_arr, G_LR, D_LR, LOSS) #this will only work after a 50 iterations to allow for proper averaging 
        # Save models
        generator.save(model_directory+'trained_3DGAN_'+'epoch_'+str(epoch)+'.h5')
  
    # Save Checkpoint after every 100 epoch
    save_checkpoints(ckpt, model_directory, epoch)

    # # Delete Model, keep last N models
    IOhousekeeping(model_directory, keep_last_N, save_freq, Restart_Training, epoch)

    # Print Status at the end of the epoch
    dt_epoch =  time.time() - t_epoch_start
    n_objects = n_objects + epoch*BATCH_SIZE
    print('\nTotal Number of objects trained: ', int(n_objects/1000), 'k', ' | time elapsed:',  str(int((time.time() - start_time) / 60.0 )), "min" , '\n')
                
    # END OF EPOCH
    ###########################################################################

    
print('\n TRAINING DONE! \n') 
print("Total Training Time: ", str((time.time() - start_time) / 60.0), "min" )