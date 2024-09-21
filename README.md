# SimultaneousPLN

This is the Python implementation of the manuscript *"Simultaneous Estimation of Many Sparse Networks via Hierarchical Poisson Log-Normal Model"*. The package requires [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) and [skggm](https://github.com/skggm/skggm) as dependencies. Please ensure that these prerequisites are properly installed.

## Installation

To install and use this package, follow these steps:

1. Download and unzip the package.
2. Navigate to the folder (e.g., `SimultaneousPLN`).
3. Build the package using the following command:

    ```bash
    python setup.py build_ext --inplace
    ```

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
from simultaneous_pln import SimultaneousPLN
```
### Prepare the Data

Prepare your count data, offset, and covariates for different groups as lists of 2D NumPy arrays, where:
- **Rows** represent samples.
- **Columns** represent features.

Ensure that the number of columns (features) is the same across all arrays within the lists. The count data and offset for each group must have the exact same dimensions.

### Model Construction

You can construct the model as follows:

```python
model = SimultaneousPLN(y, Offset, z)
```
Where:
- `y`: List of 2D arrays representing count data.
- `Offset`: List of 2D arrays representing the offset.
- `z`: List of 2D arrays representing covariates.
To initialize the model parameters, use:

```python
model.initialize()
```
If you prefer to define your own initialization for the parameters, you can pass them as arguments:
```python
model.initialize(Omega_init, mu_init, sigma_init)
```
## Fit the Model
To fit the model, simply call:
```python
model.fit()
```
You can customize the fitting preferences, such as the maximum number of iterations and multiprocessing options, by passing them as arguments. For more details, call `help(model.fit)`.

# Example
For more detailed usage and examples, please refer to example.py included in the package.

## References

--- 
bibliography: reference.bib
