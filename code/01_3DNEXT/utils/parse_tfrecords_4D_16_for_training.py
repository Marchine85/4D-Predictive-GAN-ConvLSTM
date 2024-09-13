import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def decode_object(example):    
    object4d = tf.io.decode_raw(example['object4d'], tf.uint8)
    labelBetti = tf.io.decode_raw(example['labelBetti'], tf.uint8)
    labelObjs = tf.io.decode_raw(example['labelObjs'], tf.uint8)

    object4d = tf.reshape(object4d, [128, 16, 16, 16])
    labelBetti = tf.reshape(labelBetti, [4])
    labelObjs = tf.reshape(labelObjs, [4])

    return object4d, labelBetti, labelObjs


def read_tfrecord(example):
    global NORMALIZE
    global T
    global TINC
    
    object_feature_description = {
        'object4d': tf.io.FixedLenFeature([], tf.string),
        'labelBetti': tf.io.FixedLenFeature([], tf.string),
        'labelObjs': tf.io.FixedLenFeature([], tf.string),
    }
    
    example = tf.io.parse_single_example(example, object_feature_description)

    object4d, labelBetti, labelObjs = decode_object(example)

    CROP=(128-T)//2
    # Reshape into time slices of n time steps (axis=1)
    object4d = object4d[CROP:128-CROP:TINC]
    object4d = tf.reshape(object4d, [T//TINC, 16, 16, 16])
    rearranged_objs4d = []
    for t in range(object4d.shape[0]):
        if t == object4d.shape[0] - 1:
            break
        rearranged_objs4d.append(object4d[t])
        rearranged_objs4d.append(object4d[t+1])

    object4d = tf.reshape(tf.stack(rearranged_objs4d), [T//TINC-1, 2, 16, 16, 16])

    if NORMALIZE:
        object4d = tf.cast(object4d, tf.float32) - 0.5
        
    return object4d#, tf.ones([object4d.shape[0],4], tf.uint8)*labelObjs
    #return object4d, tf.ones(object4d.shape[0], tf.uint8)*labelBetti[-1]#,  tf.ones([object4d.shape[0],4], tf.uint8)*labelObjs

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP", num_parallel_reads=AUTOTUNE)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

def get_dataset(filenames: list, shuffle: bool, batch_size: int, t: int, tinc: int, normalize: bool):
    global NORMALIZE
    global T
    global TINC
    NORMALIZE = normalize
    TINC = tinc
    T = t
    dataset = load_dataset(filenames)
    dataset = dataset.unbatch()
    if shuffle:
        dataset = dataset.shuffle(len(filenames)*(T//TINC-1)//2) # ALL: len(filenames)*(T//TINC)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    #dataset = dataset.cache()
    return dataset


# FOR TESTING:
'''
folder_name="Z:/Master_Thesis/data/4D_128_32_topological_dataset_0-31_16_label8only"

FILENAMES = tf.io.gfile.glob(folder_name+"/*.tfrecords")[:7]
split_ind = int(1 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

#TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))
#print("Test TFRecord Files:", len(TEST_FILENAMES))

BATCH_SIZE = 128

train_dataset = get_dataset(TRAINING_FILENAMES, shuffle=True, batch_size=BATCH_SIZE, t=80, tinc=4, normalize=False)
#print(len(list(train_dataset)) * BATCH_SIZE)
#object4d, labelBetti = next(iter(train_dataset))
# print(object4d.shape, labelBetti.shape)
# print(object4d[0], labelBetti)

#print(tf.data.experimental.cardinality(train_dataset))

def jaccard(x,y):
  x = np.asarray(x, bool) # Not necessary, if you keep your data
  y = np.asarray(y, bool) # in a boolean array already!
  
  #j = tf.math.reduce_sum(tf.square(differences), axis=[1,2,3,4])
  j_and = tf.math.reduce_sum(np.double(np.bitwise_and(x, y)), axis=[1,2,3])
  j_or = tf.math.reduce_sum(np.double(np.bitwise_or(x, y)), axis=[1,2,3])
  
  return  tf.reduce_mean(1 - j_and / j_or)

import dataIO
for object4d, labelBetti in train_dataset.take(1):
    print(object4d.shape, labelBetti.shape)
    jaccard_ = jaccard(object4d[:,0], object4d[:,1])

    i=0
    for object4d_i, labelBetti_i in zip(object4d, labelBetti):
        print(object4d_i.shape, labelBetti_i)
        #print(object4d_i.shape, labelBetti)

        # for object4d_ii in object4d_i:
        #     dataIO.plotFromVoxels(object4d_ii.numpy(), [i, labelBetti_i])
        #     i+=1
'''