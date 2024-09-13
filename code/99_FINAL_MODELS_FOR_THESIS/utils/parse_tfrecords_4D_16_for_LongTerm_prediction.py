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
    #object4d = tf.reshape(object4d, [128, 16, 16, 16])
    #print("Time_Points: ", object4d.shape[0])
    if NORMALIZE:
        object4d = tf.cast(object4d, tf.float32) - 0.5
        
    return object4d #, tf.ones([object4d.shape[0],4], tf.uint8)*labelObjs

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
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
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    #dataset = dataset.unbatch()
    if shuffle:
        dataset = dataset.shuffle(len(filenames)*(T//TINC))
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


# FOR TESTING:
'''
folder_name="Z:/Master_Thesis/data/4D_128_32_topological_dataset_0-31_16_label8only"

FILENAMES = tf.io.gfile.glob(folder_name+"/*.tfrecords")[:1]

print("Train TFRecord Files:", len(FILENAMES))

train_dataset = get_dataset(FILENAMES, shuffle=False, batch_size=None, t=80, tinc=4, normalize=False)

import dataIO
for object4d, labelBetti in train_dataset:
    print(object4d.shape, labelBetti)

    i=0
    for object4d_i, labelBetti_i in zip(object4d, labelBetti):
        print(object4d_i.shape, labelBetti_i.shape)
        #print(object4d_i.shape, labelBetti)

        dataIO.plotFromVoxels(object4d_i.numpy(), [i, labelBetti_i])
        i+=1
'''