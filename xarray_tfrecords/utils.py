import xarray as xr
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def feature_parse_single_sample(sample):
    """_summary_

    Args:
        sample (_type_): _description_

    Returns:
        _type_: _description_
    """
    var_feature_dict = {}
    for variable in sample.data_vars:
        var_dtype = sample[variable].dtype 
        var_feature = _get_feature_func(var_dtype)(sample[variable].values.flatten())
        var_feature_dict[variable] = var_feature
    logging.debug("feature_parse_single_sampleFeature Single Example: ")
    logging.debug(var_feature_dict)
    return var_feature_dict


def _get_tf_type(dtype):
    if dtype in (bytes,str):
        return tf.string

    if dtype in (int, np.integer, bool, np.bool_):
        return tf.int64

    if dtype in (float, np.float32):
        return tf.float32

    raise Exception(f"Unsupported type "+str(dtype))

def _get_feature_func(dtype):
    if dtype in (bytes, str):
        return _bytes_feature

    if dtype in (int, np.integer, bool, np.bool_):
        return _int64_feature

    if dtype in (float, np.float32):
        return _float_feature


    raise Exception(f"Unsupported type "+str(dtype))


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
        