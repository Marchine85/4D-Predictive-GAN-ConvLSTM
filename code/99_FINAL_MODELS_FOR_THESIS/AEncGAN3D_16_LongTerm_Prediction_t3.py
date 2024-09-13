import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
# OWN LIBRARIES
from utils import parse_tfrecords_4D_16_for_LongTerm_prediction as p4d
from utils import dataIO as d 


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

MODEL_LOCATION      = os.path.join(base_dir, "aencgan3d_16_t3_T20_TINC1")

TEST_DATASET_LOC   = "Z:/Master_Thesis/data/02_4D_test_datasets"

T                   = int(MODEL_LOCATION[-8:-6])             
TINC                = int(MODEL_LOCATION[-1])                      
N_PREDICT  	        = 5
MAX_TIME            = 5

if not os.path.exists(os.path.join(base_dir,f'objects_longterm/T{T}_TINC{TINC}')):
    os.makedirs(os.path.join(base_dir,f'objects_longterm/T{T}_TINC{TINC}'))
###############################################################################
############# GET Data/Objects ################################################
###############################################################################
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_DATASET_LOC,"*.tfrecords"))
test_set_size = len(TEST_FILENAMES)

print()
print("Total Number of TFRecord Files to Test :", test_set_size)
print()

# Convert/Decode dataset from tfrecords file for training
test_objects = p4d.get_dataset(TEST_FILENAMES, shuffle=False, batch_size=None, t=T, tinc=TINC, normalize=False)

encoder_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*encoder*.h5"))
encoder = load_model(encoder_model[-1])

generator_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*generator*.h5"))
generator = load_model(generator_model[-1])

def predictions(objects):
    n=0
    for X in objects:

        d.plotFromVoxels(np.squeeze(X[0]), f"{n}_T{T}_TINC{TINC}_t0")
        d.plotMeshFromVoxels(np.squeeze(X[0]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t0")
        d.plotFromVoxels(np.squeeze(X[1]), f"{n}_T{T}_TINC{TINC}_t1")
        d.plotMeshFromVoxels(np.squeeze(X[1]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t1")
        d.plotFromVoxels(np.squeeze(X[2]), f"{n}_T{T}_TINC{TINC}_t2")
        d.plotMeshFromVoxels(np.squeeze(X[2]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t2")
        d.plotFromVoxels(np.squeeze(X[3]), f"{n}_T{T}_TINC{TINC}_t3")
        d.plotMeshFromVoxels(np.squeeze(X[3]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t3")
        d.plotFromVoxels(np.squeeze(X[4]), f"{n}_T{T}_TINC{TINC}_t4")
        d.plotMeshFromVoxels(np.squeeze(X[4]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t4")
        d.plotFromVoxels(np.squeeze(X[5]), f"{n}_T{T}_TINC{TINC}_t5")
        d.plotMeshFromVoxels(np.squeeze(X[5]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t5")
        d.plotFromVoxels(np.squeeze(X[6]), f"{n}_T{T}_TINC{TINC}_t5")
        d.plotMeshFromVoxels(np.squeeze(X[6]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t6")
        d.plotFromVoxels(np.squeeze(X[7]), f"{n}_T{T}_TINC{TINC}_t5")
        d.plotMeshFromVoxels(np.squeeze(X[7]), obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t7")
        #Loop over time per sample
        X = tf.dtypes.cast(X, tf.float32)
        for t in range(2,X.shape[1]-3):
            if t==2:
                X_minus2 = X[0][np.newaxis,...]
                X_minus1 = X[1][np.newaxis,...]
                X0 = X[2][np.newaxis,...]
                    
            # Encode & Generate
            latent_code_tminus2 = encoder(X_minus2)
            latent_code_tminus1 = encoder(X_minus1)
            latent_code_t0 = encoder(X0)
            latent_code = tf.concat([latent_code_tminus2, latent_code_tminus1, latent_code_t0], axis=1)
            pred_object = np.squeeze(generator(latent_code, training=False))
            pred_object = np.asarray(pred_object>0.5, bool)
    
            d.plotFromVoxels(pred_object, f"{n}_T{T}_TINC{TINC}_t{t+1}_pred")
            d.plotMeshFromVoxels(pred_object, obj=f"objects_longterm/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t{t+1}_pred")
        
            if t==MAX_TIME+2-1:
                break

            X_minus2 = X_minus1
            X_minus1 = X0
            X0 = pred_object[np.newaxis,...]
            
        n+=1
        if n==N_PREDICT:
            break


predictions(test_objects)

