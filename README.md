# xarray-tfrecord converter

A collection of scripts that convert xarray to tfrecord files and back. This work was inspired by [pandas-tfrecords converter](https://github.com/schipiga/pandas-tfrecords)

## Quick start

### Installation
```python
pip install xrtfrecords
```

### Simple Example with dummy data
```python
import xarray as xr
import xarray_tfrecords
import numpy as np

data = data = np.random.normal(size=(n_time,n_lat,n_lon))
data = xr.DataArray(name = "temp", data = data, dims = ["time","lat","lon"], coords= {"time":range(n_time),"lat":range(n_lat),"lon":range(n_lon)}).to_dataset()

xarray_tfrecords.xarray_to_tfrecord(data,sample_dim="time",feature_variables = ["temp"],data_path ="test.tfrecord")

data_reconstructed = xarray_tfrecords.tfrecord_to_xarray("test.tfrecord")

xr.testing.assert_allclose(data, data_reconstructed)
```

Attributes of used Dimensions and coordinates will be reconstructed as well but so far not coordinates that are not used. Might be added later.

Metadata is stored in the same path and with the same filename and the fileextension ".meta"