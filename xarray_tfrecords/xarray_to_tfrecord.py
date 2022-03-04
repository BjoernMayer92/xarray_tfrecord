    """_summary_
    """

    
import tensorflow as tf
import pickle
import logging

class TFRecordWriterXarray(tf.io.TFRecordWriter):
    def __init__(self, path, data, feature_variables, sample_dim, options=None):
        self.path = path
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

def xarray_to_tfrecord(data, sample_dim, feature_dims, filename):
    writer = TFRecordWriterXarray(data_path, data, feature_dims, sample_dim)
    for i,sample_index in enumerate(data[sample_dim]):
        sample = data[feature_dims].sel({sample_dim:sample_index})
        sample_feature_dict = feature_parse_single_sample(sample)
        Features = tf.train.Features(feature=sample_feature_dict)
        sample_example = tf.train.Example(features = Features )

        writer.write(sample_example.SerializeToString())
        logging.info(str(i))
    writer.close()
        
        