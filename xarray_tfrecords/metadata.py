    """_summary_
    """



class metadata:
    def __init__(self, data, feature_variables, sample_dim):
        self.sample_dim = sample_dim
        self.data_vars = list(data.data_vars.keys())
        for variable in feature_variables:
            self.data_vars.remove(variable)
        self.metadata = data[self.data_vars].to_dict(data=True)
        self.feature_metadata = data[feature_variables].to_dict(data=False)