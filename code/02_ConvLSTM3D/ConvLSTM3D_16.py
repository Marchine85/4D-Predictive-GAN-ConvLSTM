import os
from shutil import copyfile
import glob
import time
import sys
import numpy as np

from keras import layers
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# OWN LIBRARIES
from utils.loss_plotter import render_graphs
from utils import dataIO as d 
from utils import parse_tfrecords_4D_16_for_training_ConvLSTM as p4d
from utils.logger import Logger

###############################################################################
#######################################
########### inputs  ###################
#######################################
N_EPOCHS            = 10000             # Default: 10000 - Max. training epochs
MAX_ITERATIONS      = 10000             # Default: -1 --> Max. Training Iterations (Max. Number of processed Batches) before training stops
TRAIN_DATASET_LOC   = "Z:/Master_Thesis/data/00_4D_training_datasets"
VALID_DATASET_LOC   = "Z:/Master_Thesis/data/01_4D_validation_datasets"
N_TRAINING          = 10000             # Number of Files for Training: -1 == USE ALL
N_VALIDATION        = 1000              # Number of Files for Validation: -1 == USE ALL
T                   = 80                # Default: 80 - Even Number of Length of Time History to use, MUST BE EVENLY DIVISIBLE BY TINC, e.g. 80 means that 80 steps out of all 128 steps are used, where 24 steps at the beginning and 24 steps at the end of the time history are cropped.
TINC                = 4                 # Default: 1 - Allowed: (1, 2, 4, 8) - e.g.: 4 --> 80/4 = 20 Timepoints --> 19 PAIRS of T0->T1 --> TIME INCREMENT FOR EXTRACTING TIMESTEPS PER DATAPOINT -   (Check parse_tfrecords_4D_##_for_training for extraction details)
NORMALIZE           = False             # NORMALIZE DATA to ZERO MEAN --> from [0;1] to [-0.5, 0.5]
LOGGING             = True              # LOG TRAINING RUN TO FILE (If Kernel get Interupted (user or error) Kernel needs to be restarted, otherwise log file is locked to be deleted!)
###############################################################################
# Restart Training?
RESTART_TRAINING    = False             
RESTART_ITERATION   = 50000             # the epoch number when the last training was stopped, has no influnce on loading the restart file, just used for tracking information numbering
RESTART_FOLDER_NAME = '20240221_012216_aencgan_3dnext_16_t3'
###############################################################################
# Output Settings
VAL_SAVE_FREQ       = 200               # Default: N_TRAINING//BATCH_SIZE - OFF: -1, every n training iterations (n batches) check validation loss
WRITE_RESTART_FREQ  = 1000              # Default: 100 - every n training iterations (n batches) save restart checkpoint
KEEP_BEST_ONLY      = True              # Default: True - Keep only the best model (KEEP_LAST_N=1)
BEST_METRIC         = "JACCARD"         # Default: JACCARD - "JACCARD" or "MSE"
# if KEEP_BEST_ONLY = False define max. Number of last models to keep:
KEEP_LAST_N         = 5                 # Default: 10 - Max. number of last files to keep
N_SAMPLES           = 5                 # Default: 10 - Number of object samples to generate after each epoch
###############################################################################
########## HYPERPARAMETERS  ###########
#######################################
BATCH_SIZE          = 16                # Default: 64 - Number of Examples per Batch / Probably Increase for better Generalization
FEATUREMAPS         = 32                # Default: 5 - Max Featuremap size (Maybe increase for learning larger/more diverse datasets...)
###############################################################################
# Optimizer Settings
LR                  = 0.001             # Default: 0.001
###############################################################################
# DEFAULT - DONT TOUCH SETTINGS
SIZE                = 16                # Default: 16 - Voxel Cube Size
###############################################################################
############# GET DATA/OBJECTS ################################################
###############################################################################
TRAINING_FILENAMES = tf.io.gfile.glob(TRAIN_DATASET_LOC+"/*.tfrecords")[:N_TRAINING]
VALIDATION_FILENAMES = tf.io.gfile.glob(VALID_DATASET_LOC+"/*.tfrecords")[:N_VALIDATION]
training_set_size = len(TRAINING_FILENAMES)
validation_set_size = len(VALIDATION_FILENAMES)
print()
print("Total Number of Samples to Train:", training_set_size)
print("Total Number of Samples for Validation:", validation_set_size)
print()

# Convert/Decode dataset from tfrecords file for training
train_objects = p4d.get_dataset(TRAINING_FILENAMES, shuffle=True, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=NORMALIZE )
val_objects = p4d.get_dataset(VALIDATION_FILENAMES, shuffle=False, batch_size=BATCH_SIZE, t=T, tinc=TINC, normalize=NORMALIZE )

###############################################################################
################### CONVOLUTIONAL LSTM NETWORK ################################
###############################################################################
# DEFINE CONVLSTM NETWORK
def ConvLSTM3D():
    inp = layers.Input(shape=(None, *(16,16,16,1))) 
    
    x = layers.ConvLSTM3D(filters=FEATUREMAPS, kernel_size=(5, 5, 5), padding="same", return_sequences=True, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM3D(filters=FEATUREMAPS, kernel_size=(3, 3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM3D(filters=FEATUREMAPS,kernel_size=(1, 1, 1), padding="same", return_sequences=True, activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(filters=1, kernel_size=(5, 5, 5), activation="sigmoid", padding="same", input_shape=x.shape[2:])(x)

    model = Model(inp, x)
    
    return model

###############################################################################
########## COMPILE NETWORK ####################################################
###############################################################################
model = ConvLSTM3D()
#Compile Model
model.compile()
# PRINT NETWORK SUMMARY
print(model.summary())

###############################################################################
############## Optimizer ######################################################
###############################################################################
optimizer = Adam(learning_rate=LR)

###############################################################################
############## LOSS ###########################################################
###############################################################################
@tf.function
def BCE_LOSS(y_true, y_pred):
    bce=tf.keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1)
    BCE_loss = tf.math.reduce_mean(bce)

    return BCE_loss

@tf.function
def MSE_LOSS(y_true, y_pred):   
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    MSE_loss = tf.reduce_mean(mse)

    return MSE_loss

def JACCARD_LOSS(X, y):    
    y_true = tf.dtypes.cast(y, tf.float32)
    y_pred = np.squeeze(model(X, training=False))
    
    y_real = np.asarray(y_true>0.5, bool)
    y_pred = np.asarray(y_pred>0.5, bool) 
            
    j_and = tf.math.reduce_sum(np.double(np.bitwise_and(y_real, y_pred)), axis=[2,3,4])
    j_or = tf.math.reduce_sum(np.double(np.bitwise_or(y_real, y_pred)), axis=[2,3,4])

    JACCARD_loss = tf.reduce_mean(np.nan_to_num(1.0 - j_and / j_or, nan=0.0))

    return JACCARD_loss

###############################################################################
############## TRAINING STEP ##################################################
###############################################################################
@tf.function
def train_step(X, y):

    X = X[...,np.newaxis]
    y_true = y[...,np.newaxis]

    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        BCE_loss = BCE_LOSS(y_true, y_pred)
        MSE_loss = MSE_LOSS(y_true, y_pred)
                
    gradients = tape.gradient(BCE_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return BCE_loss, MSE_loss

@tf.function
def validation_step(X, y):
    X = X[...,np.newaxis]
    y_true = y[...,np.newaxis]
    
    y_pred = model(X, training=False)
    BCE_loss = BCE_LOSS(y_true, y_pred)
    MSE_loss = MSE_LOSS(y_true, y_pred)
            
    return BCE_loss, MSE_loss

###############################################################################
######### SOME UTILITY FUNCTIONS ##############################################
###############################################################################
def generate_sample_objects(objects, total_iterations, tag="pred"):
    # Random Sample at every VAL_SAVE_FREQ
    i=0

    for X, y in objects:
        X = tf.dtypes.cast(X, tf.float32)
        y_pred = np.squeeze(model(X, training=False) > 0.5)
        y_pred_last = y_pred[-1]
        try:
            d.plotMeshFromVoxels(y_pred_last, obj=model_directory+tag+'_'+str(i)+'_next'+'_iter_'+str(total_iterations))
        except:
            print(f"Cannot generate STL, Marching Cubes Algo failed for sample {i}")
            # print('''Cannot generate STL, Marching Cubes Algo failed: Surface level must be within volume data range! \n
                    # This may happen at the beginning of the training, if it still happens at later stages epoch>10 --> Check Object and try to change Marching Cube Threshold.''')    

        i+=1

def IOhousekeeping(model_directory, KEEP_LAST_N, VAL_SAVE_FREQ, RESTART_TRAINING, total_iterations, KEEP_BEST_ONLY, best_iteration):
    if total_iterations > KEEP_LAST_N*VAL_SAVE_FREQ and total_iterations % VAL_SAVE_FREQ == 0:
        if KEEP_BEST_ONLY:
            fileList_all = glob.glob(model_directory + '*_iter_*')
            fileList_best = glob.glob(model_directory + '*_iter_' + str(int(best_iteration)) +'*')
            fileList_del = [ele for ele in fileList_all if ele not in fileList_best]
            
        else:
            fileList_del = glob.glob(model_directory + '*_iter_' + str(int(total_iterations-KEEP_LAST_N*VAL_SAVE_FREQ)) +'*')

        # Iterate over the list of filepaths & remove each file.
        for filePath in fileList_del:
            os.remove(filePath)

###############################################################################
########### LAST PREPARATION STEPS BEFORE TRAINING STARTS #####################
###############################################################################
#generate folders:
if RESTART_TRAINING:
    model_directory = os.getcwd() + '/' + RESTART_FOLDER_NAME + '/'
else:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    model_directory = os.getcwd()+'/'+time_string+'_' + os.path.splitext(os.path.basename(__file__))[0] + '/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)  

# Save running python file to run directory
if RESTART_TRAINING:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%Y%m%d_%H%M%S", named_tuple)
    copyfile(__file__, model_directory+time_string +'_restart_'+os.path.basename(__file__))
else:
    copyfile(__file__, model_directory+os.path.basename(__file__))

# Activate Console Logging:
if LOGGING:
    sys.stdout = Logger(filename=model_directory+'log_out.txt')

# save some real examples
i=0
train_objects_samples = []
for X, y in train_objects.take(N_SAMPLES):
    X_last = X[0][-1]
    y_last = y[0][-1]

    try:
        sample_name = model_directory+'train_'+str(i)+'_init'
        d.plotMeshFromVoxels(np.array(X_last), obj=sample_name)
        sample_name = model_directory+'train_'+str(i)+'_next'
        d.plotMeshFromVoxels(np.array(y_last), obj=sample_name)
    except:
        print("Example Object couldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")

    train_objects_samples.append([X[0][np.newaxis,...,np.newaxis], y[0][np.newaxis,...,np.newaxis]]) # for validation of reconstruction during training, see generate_sample_objects():
    i+=1    

# save some validation examples
i=0
val_objects_samples = []
for X, y in val_objects.take(N_SAMPLES):
    X_last = X[0][-1]
    y_last = y[0][-1]

    try:
        sample_name = model_directory+'valid_'+str(i)+'_init'
        d.plotMeshFromVoxels(np.array(X_last), obj=sample_name)
        sample_name = model_directory+'valid_'+str(i)+'_next'
        d.plotMeshFromVoxels(np.array(y_last), obj=sample_name)
    except:
        print("Example Object couldnt be generated with Marching Cube Algorithm. Probable Reason: Empty Input Tensor, All Zeros")

    val_objects_samples.append([X[0][np.newaxis,...,np.newaxis], y[0][np.newaxis,...,np.newaxis]]) # for validation of reconstruction during training, see generate_sample_objects():
    i+=1    
 
###############################################################################
# Define Checkpoint
checkpoint_path = model_directory+'checkpoints/'
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)

manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

def save_checkpoints(ckpt, model_directory):
    if not os.path.exists(model_directory+'checkpoints/'):
        os.makedirs(model_directory+'checkpoints/')  
    manager.save()

if RESTART_TRAINING:
    total_iterations = RESTART_ITERATION
    RESTART_EPOCH = training_set_size // BATCH_SIZE
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("\nRestored from {} \n".format(manager.latest_checkpoint))
    else:
        print("No Checkpoint found --> Initializing from scratch.")

###############################################################################
############# TRAINING ########################################################
###############################################################################
print("----------------------------------------------------------------------")
print('RUN TRAINING:\n')
###############################################################################
start_time = time.time()
iter_arr, val_iter_arr  = [], []
BCE_losses_train = []
MSE_losses_train = []
JACCARD_losses_train = []
BCE_losses_val_mean = []
MSE_losses_val_mean = []
JACCARD_losses_val_mean = []
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

    for X, y in train_objects:
        # START OF BATCH
        batch_time = time.time()   
        BCE_loss_train, MSE_loss_train = train_step(X, y)
        #Append arrays
        BCE_losses_train.append(BCE_loss_train)
        MSE_losses_train.append(MSE_loss_train)
        
        JACCARD_loss_train = JACCARD_LOSS(X, y)
        JACCARD_losses_train.append(JACCARD_loss_train)

        n_batch+=1
        total_iterations+=1
        iter_arr.append(total_iterations)    

        print("E: %3d/%5d | B: %3d/%3d | I: %5d | T: %5.1f | dt: %2.2f || LOSSES: BCE %.4f | MSE %.4f | JACCARD %.4f | " \
            % (epoch, N_EPOCHS, n_batch, training_set_size//BATCH_SIZE, total_iterations, time.time() - start_time, time.time() - batch_time,  BCE_loss_train, MSE_loss_train, JACCARD_loss_train))

        ###########################################################################
        # AFTER N TOTAL ITERATIONS GET VALIDATION LOSSES, SAVE GENERATED OBJECTS AND MODEL:
        if (total_iterations % VAL_SAVE_FREQ == 0) and VAL_SAVE_FREQ != -1:
            print('\nRUN VALIDATION, SAVE OBJECTS, MODEL AND PLOT LOSSES...')
                  
            BCE_losses_val = []            
            MSE_losses_val = []           
            JACCARD_losses_val = []        
            for X, y in val_objects:
                BCE_loss_val, MSE_loss_val = validation_step(X, y)
                BCE_losses_val.append(BCE_loss_val) # Collect Val loss per Batch
                MSE_losses_val.append(MSE_loss_val) # Collect Val loss per Batch
                
                JACCARD_loss_val = JACCARD_LOSS(X, y)
                JACCARD_losses_val.append(JACCARD_loss_val) # Collect Val loss per Batch
    
            BCE_losses_val_mean.append(np.mean(BCE_losses_val)) # Mean over batches
            MSE_losses_val_mean.append(np.mean(MSE_losses_val)) # Mean over batches
            JACCARD_losses_val_mean.append(np.mean(JACCARD_losses_val)) # Mean over batches
            
            val_iter_arr.append(total_iterations)
        ###########################################################################
            # Output generated objects
            generate_sample_objects(train_objects_samples, total_iterations, 'train')
            generate_sample_objects(val_objects_samples, total_iterations, 'valid')
            # # Plot and Save Loss Curves
            render_graphs(model_directory, BCE_losses_train, MSE_losses_train, JACCARD_losses_train, BCE_losses_val_mean, MSE_losses_val_mean, JACCARD_losses_val_mean, iter_arr, val_iter_arr, RESTART_TRAINING) #this will only work after a 50 iterations to allow for proper averaging 

            if KEEP_BEST_ONLY:
                if BEST_METRIC == "JACCARD":
                    BEST_LOSS_METRIC = JACCARD_losses_val_mean
                    KEEP_LAST_N = 1
                elif BEST_METRIC == "MSE":
                    BEST_LOSS_METRIC = MSE_losses_val_mean
                    KEEP_LAST_N = 1
                if len(BEST_LOSS_METRIC) > 1 and (BEST_LOSS_METRIC[-1] <= best_loss):
                    best_loss = BEST_LOSS_METRIC[-1]
                    best_iteration = total_iterations
                    # Save models
                    model.save(model_directory+'_trained_ConvLSTM3D_'+'iter_'+str(total_iterations)+'.h5')
            else:
                # Save models
                model.save(model_directory+'_trained_ConvLSTM3D_'+'iter_'+str(total_iterations)+'.h5')
                
            # # Delete Model, keep best or last N models
            IOhousekeeping(model_directory, KEEP_LAST_N, VAL_SAVE_FREQ, RESTART_TRAINING, total_iterations, KEEP_BEST_ONLY, best_iteration)  
            
        ###########################################################################
        # SAVE CHECKPOINT FOR RESTART:
        if total_iterations % WRITE_RESTART_FREQ == 0:
            print('WRITE RESTART FILE...\n')
            # Save Checkpoint after every 100 epoch
            save_checkpoints(ckpt, model_directory)
       
        ################################  
        # STOP TRAINING AFTER MAX DEFINED ITERATIONS ARE REACHED
        if total_iterations == MAX_ITERATIONS :
            break
    if total_iterations == MAX_ITERATIONS:
        break    
    
    # Print Status at the end of the epoch
    dt_epoch =  time.time() - t_epoch_start
    n_objects = n_objects + training_set_size
    print("\n--------------------------------------------------------------------")
    print('END OF EPOCH ', epoch,' | Total Training Iterations: ', int(total_iterations), ' | Total Number of objects trained: ', int(n_objects/1000), 'k', ' | time elapsed:',  str(int((time.time() - start_time) / 60.0 )), "min")
    print("--------------------------------------------------------------------\n\n")

    # END OF EPOCH
    ###########################################################################

    
print('\n TRAINING DONE! \n') 
print("Total Training Time: ", str((time.time() - start_time) / 60.0), "min" )