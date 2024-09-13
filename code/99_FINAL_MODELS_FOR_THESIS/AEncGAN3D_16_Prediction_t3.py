import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
# OWN LIBRARIES
from utils import parse_tfrecords_4D_16_t3_for_training as p4d
from utils import dataIO as d 


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

MODEL_LOCATION      = os.path.join(base_dir, "aencgan3d_16_t3_T80_TINC4")

TEST_DATASET_LOC   = "Z:/Master_Thesis/data/02_4D_test_datasets"

T                   = int(MODEL_LOCATION[-8:-6])             
TINC                = int(MODEL_LOCATION[-1])                      
N_PREDICT  	        = 5

if not os.path.exists(os.path.join(base_dir,f'objects/T{T}_TINC{TINC}')):
    os.makedirs(os.path.join(base_dir,f'objects/T{T}_TINC{TINC}'))
###############################################################################
############# GET Data/Objects ################################################
###############################################################################
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_DATASET_LOC,"*.tfrecords"))
test_set_size = len(TEST_FILENAMES)

print()
print("Total Number of TFRecord Files to Test :", test_set_size)
print()

# Convert/Decode dataset from tfrecords file for training
test_objects = p4d.get_dataset(TEST_FILENAMES, shuffle=True, batch_size=1, t=T, tinc=TINC, normalize=False)

encoder_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*encoder*.h5"))
encoder = load_model(encoder_model[-1])

generator_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*generator*.h5"))
generator = load_model(generator_model[-1])

def predictions(objects):
    n=0
    for X in objects:
        d.plotFromVoxels(np.squeeze(X[0,0]), f"{n}_T{T}_TINC{TINC}_t-1")
        d.plotMeshFromVoxels(np.squeeze(X[0,0]), obj=f"objects/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t0")
        d.plotFromVoxels(np.squeeze(X[0,1]), f"{n}_T{T}_TINC{TINC}_t-2")
        d.plotMeshFromVoxels(np.squeeze(X[0,1]), obj=f"objects/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t1")
        d.plotFromVoxels(np.squeeze(X[0,2]), f"{n}_T{T}_TINC{TINC}_t0")
        d.plotMeshFromVoxels(np.squeeze(X[0,2]), obj=f"objects/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t2")
        d.plotFromVoxels(np.squeeze(X[0,3]), f"{n}_T{T}_TINC{TINC}_t1")
        d.plotMeshFromVoxels(np.squeeze(X[0,3]), obj=f"objects/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t3")
                
        # Encode & Generate        
        X0 = tf.dtypes.cast(X[:,0], tf.float32)
        
        Xminus2 = tf.dtypes.cast(X[:,0], tf.float32)
        Xminus1 = tf.dtypes.cast(X[:,1], tf.float32)
        X0 = tf.dtypes.cast(X[:,2], tf.float32)
        latent_code_tminus2 = encoder(Xminus2)
        latent_code_tminus1 = encoder(Xminus1)
        latent_code_t0 = encoder(X0)
        latent_code = tf.concat([latent_code_tminus2, latent_code_tminus1, latent_code_t0], axis=1)
        pred_object = np.squeeze(generator(latent_code, training=False))
        pred_object = np.asarray(pred_object>0.5, bool)

        d.plotFromVoxels(pred_object, f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(pred_object, obj=f"objects/T{T}_TINC{TINC}/multiple_{n}_T{T}_TINC{TINC}_t3_pred")
        n+=1
        if n==N_PREDICT:
            break


predictions(test_objects)

