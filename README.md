# SimultaneousPLN<img src="src/icon.svg" align="right" width="155"/>

This is the Python implementation of the manuscript *"Simultaneous Estimation of Many Sparse Networks via Hierarchical Poisson Log-Normal Model"*. 

## Installation

To install and use this package, follow these steps:

1. Download and unzip the package.
2. Navigate to the folder (e.g., `SimultaneousPLN`).
3. Build the package using the following command:

    ```bash
    python setup.py build_ext --inplace
    ```

Note: this package requires `openmp` for multithread processing, which is not supported on apple's `llvm`. If you're using macOS, you may install brew's `llvm` and set the corresponding reference before installing this package:
```bash
brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
```

A [docker](https://www.docker.com/) container of jupyter notebook can be constructed from the `Dockerfile`.

### Adding to System PATH

If you want to add the SimultaneousPLN package to your system `PYTHONPATH`, run:

```bash
export PYTHONPATH=/path/to/SimultaneousPLN:$PYTHONPATH
```
Make sure to replace `/path/to/SimultaneousPLN` with the actual path to your SimultaneousPLN folder.

## Usage
### Import the package
To use the SimultaneousPLN package, import it as follows:
```python
from simultaneous_pln import SimultaneousPLN as spln
```
### Prepare the Data

Prepare your count data, offset, and covariates for different groups as lists of 2D NumPy arrays, where:
- **Rows** represent samples.
- **Columns** represent features.

Ensure that the number of columns (features) is the same across all arrays within the lists. The count data and offset for each group must have the exact same dimensions.

### Model Construction

You can construct the model as follows:

```python
model = spln(y, Offset, z)
```
Where:
- `y`: List of 2D arrays representing count data.
- `Offset`: List of 2D arrays representing the offset.
- `z`: List of 2D arrays representing covariates.

The model will be initiaialized accordingly. You can also indlude the initialization of model parameters if you prefer to define them by yourself.

## Fit the Model
To fit the model, simply call:
```python
model.fit()
```
You can customize the fitting preferences, such as the maximum number of iterations and multiprocessing options, by passing them as arguments. For more details, call `help(model.fit)`.

For hyperparameter determination and model selection by AIC, BIC and EBIC measurements, you may use:
```python
model.ModelSelect()
```

# Example
For more detailed usage and examples, please refer to example.py included in the package.

# References

--- 
>Changhao Ge, Hongzhe Li, 2024. _Simultaneous estimation of many sparse networks via hierarchical poisson log-normal model._ [arxiv.org/abs/2409.12275](https://arxiv.org/abs/2409.12275)
