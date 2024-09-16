# SimultaneousPLN
This is the python implementation of the manuscript "Simultaneous Estimation of Many Sparse Networks via Hierarchical Poisson Log-Normal Model".

## Install
To use this package, download the zip file, unzip and navigate to the folder (for example, SimultaneousPLN). Run the setup.py file to build this package:

```shell
python setup.py build_ext --inplace
```
If you want to add the SimultaneousPLN package into system PATH, run:

```shell
export PYTHONPATH=/path/to/SimultaneousPLN:$PYTHONPATH
```
replace the `/path/to/SimultaneousPLN` with your actual path of the SimultaneousPLN folder.

## Use the package

Import the package by:
```python
from simultaneous_pln import SimultaneousPLN
```

Prepare your count data, offset and covariates of different groups in a list of 2d numpy arrays, with rows representing samples and columns being features. For each list of data, the number of columns (features) of all arrays should be the same. The count data and offset of each group should also have the exact dimensions.

Construct the model by:
```python
model = SimultaneousPLN(y, Offset, z)
```

Here, `y`, `Offset` and `z` are the count data, offset and covariates, respectively. We initialize the model parameters by:
```python
model.initialize()
```
If user prefer self-defined parameters initialization, one can pass it to model arguments:

```python
model.initialize(Omega_init, mu_init, sigma_init)
```

Fit the model by:
```python
model.fit()
```

For more package details, please refer to `example.py` file.
