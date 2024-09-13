# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:23:46 2023

@author: mgold
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def decode_object(example):    
    object4d = tf.io.decode_raw(example['object4d'], tf.uint8)
    labelBetti = tf.io.decode_raw(example['labelBetti'], tf.uint8)
    labelObjs = tf.io.decode_raw(example['labelObjs'], tf.uint8)

    object4d = tf.reshape(object4d, [128, 32, 32, 32])
    labelBetti = tf.reshape(labelBetti, [4])
    labelObjs = tf.reshape(labelObjs, [4])

    return object4d, labelBetti, labelObjs


def read_tfrecord(example):
    object_feature_description = {
        'object4d': tf.io.FixedLenFeature([], tf.string),
        'labelBetti': tf.io.FixedLenFeature([], tf.string),
        'labelObjs': tf.io.FixedLenFeature([], tf.string),
    }
    
    example = tf.io.parse_single_example(example, object_feature_description)

    object4d, labelBetti, labelObjs = decode_object(example)
    
    # 3D GAN PREPARATION OF 4D Data:
    # Take only the middle time instant (128/2 = 64)
    # Simplified Label Betti, take only last Label (sum of objects)
    # Label Objects NOT USED!
    object4d, labelBetti, _ = decode_object(example)
    object4d = object4d[64]
    labelBetti = labelBetti[-1]

    return object4d, labelBetti

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP", buffer_size=10000000, num_parallel_reads=AUTOTUNE)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset

def get_dataset(filenames: list, shuffle: True, batch_size: int):
    dataset = load_dataset(filenames)
    if shuffle:
        dataset = dataset.shuffle(len(filenames))
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
