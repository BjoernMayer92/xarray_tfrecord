import numpy as np


def gen_feature_description(metadata_instance, sample_dim):
    feature_variables = metadata_instance.feature_metadata["data_vars"]
    feature_description = {}
    for feature_variable in feature_variables:
        feature_variable_dimensions = metadata_instance.feature_metadata["data_vars"][feature_variable]["dims"]
        feature_variable_dimensions = list(feature_variable_dimensions)
        feature_variable_dimensions.remove(sample_dim)
        feature_variable_dimension_shape = []
        for feature_variable_dimension in feature_variable_dimensions:
            dimension_size = metadata_instance.metadata["dims"][feature_variable_dimension]
            feature_variable_dimension_shape.append(dimension_size)
        
        dtype = np.dtype(metadata_instance.feature_metadata["data_vars"][feature_variable]["dtype"])
        tf_type = _get_tf_type(dtype = dtype)
        
        feature_description[feature_variable] = tf.io.FixedLenFeature(feature_variable_dimension_shape, tf_type)
    return feature_description
    

def _parser_function(example_proto, feature_description):
    return tf.io.parse_single_example(example_proto, feature_description)

def _get_parser_function(feature_description):
    return lambda example_proto : _parser_function(example_proto, feature_description)
    

def _load_variable_from_parsed_dataset(dataset, variable):
    """_summary_

    Args:
        dataset (_type_): _description_
        variable (_type_): _description_
    """
    
    return np.stack([sample[variable].numpy() for sample in dataset])



def tfrecord_to_xarray(data_path):
    dataset = tf.data.TFRecordDataset(data_path)
    metadata_filename = data_path.split(".")[0]+".meta"

    with open(metadata_filename,"rb") as handle:
        metadata_instance = pickle.load(handle)

    feature_description = gen_feature_description(metadata_instance, sample_dim="time")
    dataset_parsed = dataset.map( _get_parser_function(feature_description))


    variable_arr = []
    metadata_coords = metadata_instance.metadata["coords"]
    for variable in metadata_instance.feature_metadata["data_vars"]:
        data = _load_variable_from_parsed_dataset(dataset_parsed, variable=variable)
        
        sample_arr = []
        metadata_variable = metadata_instance.feature_metadata["data_vars"]
        for sample in dataset_parsed:
            sample_arr.append(sample["tos"].numpy())

        variable_dimensions = metadata_variable[variable]["dims"]
        variable_attributes = metadata_variable[variable]["attrs"]

        dimension_dict = {}
        dimension_attr = {}

        for dimension in variable_dimensions:
            dimension_dict[dimension] = metadata_coords[dimension]["data"]
            dimension_attr[dimension] = metadata_coords[dimension]["attrs"]
        
        data_xr = xr.DataArray(data, name = variable, dims = variable_dimensions, coords = dimension_dict, attrs = metadata_instance.metadata["attrs"] )
        
        # Add Attributes
        for dimension in variable_dimensions:
            data_xr[dimension].attrs =  dimension_attr[dimension]
        
        data_xr = data_xr.to_dataset()[variable]
        data_xr.attrs = variable_attributes
        variable_arr.append(data_xr)
        # Add Attributes
        #data = data.assign_attrs(metadata_instance.metadata["attrs"])

    return variable_arr