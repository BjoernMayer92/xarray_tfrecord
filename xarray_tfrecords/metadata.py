import os
import pickle

class metadata:
    def __init__(self, data, feature_variables, sample_dim):
        self.sample_dim = sample_dim
        self.data_vars = list(data.data_vars.keys())
        #for variable in feature_variables:
        #    self.data_vars.remove(variable)
        self.metadata = data[data.dims.keys()].to_dict(data=True)
        self.feature_metadata = data[feature_variables].to_dict(data=False)


def load_metadata(tfrecord_filename):
    """ Loads the metadate belonging to a given tfrecord filename

    Args:
        tfrecord_filename (_type_): _description_

    Returns:
        metadata: metadata associated with tfrecord file
    """
    filename = os.path.abspath(tfrecord_filename)
    filename_meta = filename.split(".")[0] +".meta"
    with open(filename_meta,"rb") as handle:
        metadata_instance = pickle.load(handle)
    return metadata_instance