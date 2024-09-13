import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
# OWN LIBRARIES
from utils import parse_tfrecords_4D_16_for_training_ConvLSTM as p4d
from utils import dataIO as d 


base_dir = r"Z:\Master_Thesis\code\99_FINAL_MODELS_FOR_THESIS"

MODEL_LOCATION      = os.path.join(base_dir, "convlstm3d_16_T20_TINC1")

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

convlstm3d_model = tf.io.gfile.glob(os.path.join(MODEL_LOCATION, "*ConvLSTM3D*.h5"))
convlstm3d = load_model(convlstm3d_model[-1])

def predictions(objects):
    n=0
    for X, y in objects:
        d.plotFromVoxels(np.squeeze(X[0,0]), f"{n}_T{T}_TINC{TINC}_t-1")
        d.plotMeshFromVoxels(np.squeeze(X[0,0]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t0")
        d.plotFromVoxels(np.squeeze(X[0,1]), f"{n}_T{T}_TINC{TINC}_t-2")
        d.plotMeshFromVoxels(np.squeeze(X[0,1]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t1")
        d.plotFromVoxels(np.squeeze(X[0,2]), f"{n}_T{T}_TINC{TINC}_t0")
        d.plotMeshFromVoxels(np.squeeze(X[0,2]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t2")
        d.plotFromVoxels(np.squeeze(X[0,3]), f"{n}_T{T}_TINC{TINC}_t1")
        d.plotMeshFromVoxels(np.squeeze(X[0,3]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t3")
        d.plotFromVoxels(np.squeeze(X[0,4]), f"{n}_T{T}_TINC{TINC}_t1")
        d.plotMeshFromVoxels(np.squeeze(X[0,4]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t4")
        d.plotFromVoxels(np.squeeze(X[0,5]), f"{n}_T{T}_TINC{TINC}_t1")
        d.plotMeshFromVoxels(np.squeeze(X[0,5]), obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t5")
                
        # Encode & Generate        
        X = tf.dtypes.cast(X[:,0:5], tf.float32)
        y_pred = np.squeeze(convlstm3d(X[...,np.newaxis], training=False))
        y_pred = np.asarray(y_pred>0.5, bool) 

        d.plotFromVoxels(y_pred[0], f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(y_pred[0], obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotFromVoxels(y_pred[1], f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(y_pred[1], obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t2_pred")
        d.plotFromVoxels(y_pred[2], f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(y_pred[2], obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t3_pred")
        d.plotFromVoxels(y_pred[3], f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(y_pred[3], obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t4_pred")
        d.plotFromVoxels(y_pred[4], f"{n}_T{T}_TINC{TINC}_t1_pred")
        d.plotMeshFromVoxels(y_pred[4], obj=f"objects/T{T}_TINC{TINC}/convlstm_{n}_T{T}_TINC{TINC}_t5_pred")
        n+=1
        if n==N_PREDICT:
            break


predictions(test_objects)

