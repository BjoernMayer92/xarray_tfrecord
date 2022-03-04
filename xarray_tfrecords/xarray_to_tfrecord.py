from .utils import *
from .metadata import *

import tensorflow as tf
import pickle
import logging
import os

class TFRecordWriterXarray(tf.io.TFRecordWriter):
    def __init__(self, path, data, feature_variables, sample_dim, options=None):
        self.path = os.path.abspath(path)
        self.filename = self.path.split(".")[0]
        self.metadata = metadata(data, feature_variables, sample_dim)
        
        tf.io.TFRecordWriter.__init__(self,path, options)

    def write(self,record):
        tf.io.TFRecordWriter.write(self,record)

    def close(self):
        tf.io.TFRecordWriter.close(self)
        self.writeMetadata()

    def writeMetadata(self):
        self.filename_metadata = self.filename+".meta"
        
        with open(self.filename_metadata,"wb") as handle:
            pickle.dump(self.metadata, handle)

def xarray_to_tfrecord(data, sample_dim, feature_variables, data_path):
    """Transforms and saves an xarray Dataset into a tensorflow record file 

    Args:
        data (xarray.Dataset): Dataset that should be saved
        sample_dim (str): Dimension of the dataset that is used as the sample dimension
        feature_variables (list of str): Name of all feature variables that are stored
        data_path (str): Path of the output file
    """
    writer = TFRecordWriterXarray(data_path, data, feature_variables, sample_dim)
    for i,sample_index in enumerate(data[sample_dim]):
        sample = data[feature_variables].sel({sample_dim:sample_index})
        sample_feature_dict = feature_parse_single_sample(sample)
        Features = tf.train.Features(feature=sample_feature_dict)
        sample_example = tf.train.Example(features = Features )

        writer.write(sample_example.SerializeToString())
        logging.info(str(i))
    writer.close()
        
        