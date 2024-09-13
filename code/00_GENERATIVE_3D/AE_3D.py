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
from loss_plotter import render_graphs_vae
import dataIO as d
import parse_tfrecords_3D_for_training as p3d

np.set_printoptions(suppress=True,linewidth=np.nan)

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
BASE_SIZE           = 4                 # Base Resolution to start from for decoder and to end with for encoder
FEATUREMAP_BASE_DIM = 64                # Max Featuremap size (Maybe increase for learning larger/more diverse datasets...)
KERNEL_SIZE         = 5                 # Default: 5 -Lower values learn pretty bad, higher than 5 doesnt change training much
TRAIN_VAL_RATIO     = 0.05               # Fraction of objects of full dataset to be used for training (use 0.0005 to just draw 1 sample for training)
DATASET_LOCATION    = "D:/Master_Thesis/data/4D_128_32_topological_dataset_0-31_32"
N_LABELS            = 16 
###############################################################################
# Restart Training?
Restart_Training    = False
restart_epoch       = 1000
restart_folder_name = '20230415_110332_multiple_dataset_RasGANloss'
###############################################################################
# Output Settings
val_freq            = 10                # Default: 10 - OFF: -1, every n epochs check validation loss
save_freq           = 10                # Default: 10 - every n epochs print examples, save model, plot losses, do evalution of validation set
keep_last_N         = 10                # Default: 10 - Max. number of last files to keep
n_samples           = 10                # Default: 10 - Number of object samples to generate after each epoch
###############################################################################
# Define Decoder/Encoder Architecture
AE_ARCHITECTURE    = "Classic"          # Default: "PreRes" - "PreRes" or "Classic" Convolutions
###############################################################################
# Define LOSS    
LOSS                = 'SIGMOID'         # Default: SIGMOID - Choose either "SIGMOID" (CROSS-ENTROPY LOSS) or "MSE" (MEAN-SQUARED-ERROR))
###############################################################################
# Optimizer Settings
Encoder_LR          = 0.001            # Default: 0.0002 (WGAN-GP & WGAN-GP-ADA) - Decoder Learning Rate
Decoder_LR          = Encoder_LR        # Default: 0.0002 (WGAN-GP & WGAN-GP-ADA) - Encoder Learning Rate
###############################################################################
# Dropout
DROPOUT             = True              # Default: True - Use Dropout inside Encoder
# if dropout True, define Dropout Rate
DROPOUT_RATE        = 0.2               # Default: 0.2
###################
# Set BatchNorm  
BATCHNORM           = True              # Default: True - Use Batchnorm in Encoder/Decoder
###############################################################################
# Global Seed for Randomization
tf.random.set_seed(None)
###############################################################################
# weight_initializer
weight_initializer = 'glorot_uniform'   # "glorot_uniform" or "he_uniform"
####################################################################################################################
################### NETWORKS ##################################################
###############################################################################
# label input
con_label = Input(shape=(1,))
# latent vector input
latent_vector = Input(shape=(Z_SIZE,))
###############################################################################
# DEFINE Conditional Layers for Decoder
def label_conditioned_decoder(N_LABELS=N_LABELS, embedding_dim=10):
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
# Decoder PreResidual Block
def DecoderPreResBlock(x, in_filters, out_filters, kernel_size, up_size, kernel_initializer, padding, use_bias):
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
# DEFINE DECODER NETWORK
def ClassicDecoder():
    label_output = label_conditioned_decoder()
    latent_vector_output  = latent_input()
    
    # merge label_conditioned_decoder and latent_input output
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

    out_layer = x

    model = Model([latent_vector, con_label], out_layer)
    
    return model

def PreResDecoder():
    label_output = label_conditioned_decoder()
    latent_vector_output = latent_input()
    
    # merge label_conditioned_decoder and latent_input output
    x = Concatenate()([latent_vector_output, label_output])
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # PreRes Blocks
    x = DecoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM,    out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)
    x = DecoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)
    x = DecoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, up_size=2, kernel_initializer = weight_initializer, padding='same', use_bias=False)

    # Final Convolution
    x = Conv3DTranspose(filters=1, kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same')(x)

    # End Activation
    out_layer = x

    model = Model([latent_vector, con_label], out_layer)
    
    return model

###############################################################################
# DEFINE Conditional Layers for Encoder
def label_conditioned_encoder(object_dim=(SIZE, SIZE, SIZE, 1), N_LABELS=N_LABELS, embedding_dim=10):
   
    # embedding for categorical input
    label_embedding = Embedding(N_LABELS, embedding_dim)(con_label)
    #print(label_embedding)
    # linear multiplication
    nodes = SIZE*SIZE*SIZE
    label_dense = Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = Reshape((SIZE, SIZE, SIZE, 1))(label_dense)
    return label_reshape_layer

def obj_input_encoder(object_dim=(SIZE, SIZE, SIZE, 1)):
    obj_in = Input(shape=object_dim)
    return obj_in
###############################################################################
# Encoder PreResidual Block
def EncoderPreResBlock(x, in_filters, out_filters, kernel_size, strides, kernel_initializer, padding):
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
# DEFINE ENCODER NETWORK
def ClassicEncoder():
    label_conditioned_encoder_in = label_conditioned_encoder()
    obj_in = obj_input_encoder()
    
    # merge label_conditioned_decoder and latent_input output
    x = Concatenate()([obj_in, label_conditioned_encoder_in])

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
    
    out_layer = Dense(Z_SIZE, activation=None)(x)

    # define model
    model = Model([obj_in, con_label], out_layer)
    return model

def PreResEncoder():
    label_conditioned_encoder_in = label_conditioned_encoder()
    obj_in = obj_input_encoder()
    
    # merge label_conditioned_encoder and latent_input output
    x = Concatenate()([obj_in, label_conditioned_encoder_in])

    # Begin Activation
    x = Conv3D(filters=FEATUREMAP_BASE_DIM//8, kernel_size=KERNEL_SIZE, strides=2, input_shape=[SIZE, SIZE, SIZE, 1], kernel_initializer = weight_initializer, padding='same')(x)
    if BATCHNORM:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)
    
    # PreRes Blocks
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//8, out_filters=FEATUREMAP_BASE_DIM//4, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//4, out_filters=FEATUREMAP_BASE_DIM//2, kernel_size=KERNEL_SIZE, strides=2, kernel_initializer = weight_initializer, padding='same')
    x = EncoderPreResBlock(x, in_filters=FEATUREMAP_BASE_DIM//2, out_filters=FEATUREMAP_BASE_DIM,    kernel_size=KERNEL_SIZE, strides=1, kernel_initializer = weight_initializer, padding='same')
    
    # Flatten (reshape) and Linear (dense) Layer
    x = Flatten()(x)
    if DROPOUT:
        x = Dropout(DROPOUT_RATE)(x)
    out_layer = Dense(Z_SIZE, activation=None)(x)

    # define model
    model = Model([obj_in, con_label], out_layer)
    return model

#############################################################################################################################################################
###############################################################################
########## COMPILE NETWORKS ###################################################
###############################################################################
if AE_ARCHITECTURE == "PreRes":
    decoder, encoder = PreResDecoder(), PreResEncoder()
elif AE_ARCHITECTURE == "Classic":
    decoder, encoder = ClassicDecoder(), ClassicEncoder()
else:
    ValueError("Decoder/Encoder Architecture not Available. Choose between 'PreRes' or 'Classic'!")
decoder.compile()  
print(decoder.summary())
encoder.compile()  
print(encoder.summary())

##############################################################################################################################################################
###############################################################################
############## Optimization ###################################################
###############################################################################

encoder_optimizer = Adam(learning_rate=Encoder_LR)
decoder_optimizer = Adam(learning_rate=Decoder_LR)

    
encoder_vars = encoder.trainable_variables
decoder_vars = decoder.trainable_variables

print('\nDecoder-vars count= ' + str(len(decoder_vars)))
print('Encoder-vars count= ' + str(len(encoder_vars)))

##############################################################################################################################################################
###############################################################################
############## LOSS Function ##################################################
###############################################################################
#------------------------------------------------------------------------------
# LOSS
@tf.function
def compute_loss(real_objects, real_labels):

    real_objects = tf.dtypes.cast(real_objects, tf.float32)
    # Encode
    latent_code = encoder([real_objects, real_labels], training=True)
    # Decode
    logits_objects = decoder([latent_code, real_labels], training=True)   
    
    if LOSS == "MSE":
        # MSE LOSS
        # Manually Calc MSE Loss for 3D Objects
        differences = logits_objects - real_objects
        squared_sum = tf.math.reduce_sum(tf.square(differences), axis=[1,2,3,4])
        # MSE for whole batch
        loss = tf.reduce_mean(squared_sum)
        
    elif LOSS == "SIGMOID":
        # CROSS ENTROPY LOSS
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_objects, labels=real_objects)
        loss = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3, 4]))


    return loss

####################################################################################################################
####################################################################################################################
###############################################################################
# SOME UTILITY FUNCTIONS
def generate_sample_objects(test_objects, epoch):
    # Random Sample at every save_freq
    i=0
   
    for X, y in test_objects:
        X = tf.dtypes.cast(X, tf.float32)
        # Encode
        latent_code = encoder([X, y], training=False)
        # Decode
        if LOSS == "SIGMOID":
            sample_object = sigmoid(decoder([latent_code, y], training=False))
        elif LOSS == "MSE":
            sample_object = decoder([latent_code, y], training=False)

        object_out = np.squeeze(sample_object>0.5)

        try:
            #np.save(model_directory + 'np_' + str(epoch) +'_'+str(i), objects_out)
            d.plotMeshFromVoxels(object_out, obj=model_directory+'sample_epoch_'+str(epoch)+'_label_'+str(y.numpy()[0])+'_'+str(i+1))
        except:
            print(f"Cannot generate STL, Marching Cubes Algo failed for sample {i}")
            # print('''Cannot generate STL, Marching Cubes Algo failed: Surface level must be within volume data range! \n
                    # This may happen at the beginning of the training, if it still happens at later stages epoch>10 --> Check Object and try to change Marching Cube Threshold.''')    
        i+=1

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
@tf.function
def train_step(obj_batch, label_batch):
    real_objects = obj_batch[...,np.newaxis]
    real_labels = label_batch[...,np.newaxis]
    
    with tf.GradientTape() as decoder_tape, tf.GradientTape() as encoder_tape:
        loss = compute_loss(real_objects, real_labels)

    gradients_of_decoder = decoder_tape.gradient(loss, decoder.trainable_variables)
    gradients_of_encoder = encoder_tape.gradient(loss, encoder.trainable_variables)

    decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))

    return loss

@tf.function
def validation_step(val_obj_batch, val_label_batch):
    real_objects = val_obj_batch[...,np.newaxis]
    real_labels = val_label_batch[...,np.newaxis]

    val_loss = compute_loss(real_objects, real_labels)
    
    return val_loss

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
    model_directory = os.getcwd()+'/'+time_string+'_AE' + '/'
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
test_objects = []
test_labels = []
for X, y in objects.unbatch().take(n_samples):
    #print('X:', X.shape, 'y:', y.numpy())
    try:
        sample_name = model_directory+'label_'+str(y.numpy())+'_'+str(i)+'ex'
        d.plotMeshFromVoxels(np.array(X), obj=sample_name)
    except:
        print("Example Object couldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")
    test_objects.append([X[np.newaxis,...,np.newaxis], y[...,np.newaxis]]) # for validation of reconstruction during training, see generate_sample_objects():
    i+=1    

if Restart_Training:
    restart_epoch = restart_epoch
else:
    restart_epoch = 0

###############################################################################
# Define Checkpoint
checkpoint_path = model_directory+'checkpoints/'
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt = tf.train.Checkpoint(decoder_optimizer=decoder_optimizer,
                                 encoder_optimizer=encoder_optimizer,
                                 decoder=decoder,
                                 encoder=encoder)

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
train_loss, val_loss, epoch_arr, val_epoch_arr = [], [], [], []
n_objects = 0
for epoch in range(1+restart_epoch, 1+N_EPOCHS):
    ###########################################################################
    # START OF EPOCH
    t_epoch_start = time.time()
    n_batch=1
    # TRAIN ON BATCH
    Train_loss = [] 
    for obj_batch, label_batch in objects:
        ##################################
        # START OF BATCH
        batch_time = time.time()
        
        tloss = train_step(obj_batch, label_batch)
        #Append arrays
        Train_loss.append(tloss.numpy())

        print("Epoch: [%2d/%2d] Batch: [%4d/%4d] elapsed_time: %.1f, dt_batch: %.2f, train_loss: %.4f" \
            % (epoch, N_EPOCHS, n_batch, training_set_size//BATCH_SIZE, time.time() - start_time, time.time() - batch_time, tloss))

        n_batch+=1            
        # END OF BATCH
        ################################
    train_loss.append(np.mean(Train_loss))    
    epoch_arr.append(epoch)        
    
    ###########################################################################
    # Get Validation Loss
    if (epoch % val_freq == 0) and val_freq != -1:
        Val_loss = []            
        for val_obj_batch, val_label_batch in val_objects:
            vloss = validation_step(val_obj_batch, val_label_batch)
            Val_loss.append(vloss.numpy()) # Collect Val loss per Batch
        val_loss.append(np.mean(Val_loss)) # Mean over batches
        val_epoch_arr.append(epoch)
    ###########################################################################
    # AFTER EACH EPOCH TRAINING IS DONE:
    if epoch % save_freq == 0:
        # output generated objects
        generate_sample_objects(test_objects, epoch)
        # Plot and Save Loss Curves
        render_graphs_vae(model_directory, epoch, train_loss, val_loss, epoch_arr, val_epoch_arr, Decoder_LR, Encoder_LR, "MSE") #this will only work after a 50 iterations to allow for proper averaging 
        # Save models
        decoder.save(model_directory+'trained_AE_epoch_'+str(epoch)+'.h5')
  
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