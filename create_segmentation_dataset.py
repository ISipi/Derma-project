"""
Use TensorFlow helper functions to create training, validation and testing datasets from derma_data.csv

"""


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
data_loc = pd.read_csv("derma_data.csv")

print(data_loc)



# tf helper functions to create datasets
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# serialize example
def serialization(image, segment, image_filename, segment_filename, label):
    feature = {
        'image': _bytes_feature(image),
        'segment': _bytes_feature(segment),
        'image_filename': _bytes_feature(image_filename),
        'segment_filename': _bytes_feature(segment_filename),
        'label': _float_feature(label),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto


# create a tfrecord
def create_tf_record(filename_label_tuple, train_or_test):
    """
    TODO: change the file save location (record_file variable)
    :param filename_label_tuple:
    :param train_or_test:
    :return:
    """
    record_file = f'{train_or_test}.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for image, segment, label in filename_label_tuple:
            image_bytes = open(image, 'rb').read()
            segment_bytes = open(segment, 'rb').read()

            proto_example = serialization(image_bytes, segment_bytes, image, segment, label)
            writer.write(proto_example.SerializeToString())


# divide data into train and test sets, with train further divided into 10 folds for cross-validation

training_set, test_set = train_test_split(data_loc, test_size=0.1)  # take 10% of the dataset for testing
skf = StratifiedKFold(n_splits=10)  # make the 10 splits

# create the training folds:
folds = [(train, test) for train, test in skf.split(training_set, training_set['Diagnosis'])]


# loop through the training and testing sets and feed them to create_tf_record, which writes them to file
for fold in folds:
    for i in fold:
        print(len(i))

#for row in zip(data_loc["Ground truth"], data_loc['Segment']):
    #print(row)
