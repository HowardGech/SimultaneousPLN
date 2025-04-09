# SimultaneousPLN<img src="src/icon.svg" align="right" width="155"/>

This is the Python implementation of the manuscript *"Simultaneous Estimation of Many Sparse Networks via Hierarchical Poisson Log-Normal Model"*. 

## Installation

To install and use this package, follow these steps:

1. Download and unzip the package, or clone the git repo.
2. Navigate to the folder (e.g., `SimultaneousPLN`).
3. Build the package using the following command:

    ```bash
    pip install .
    ```

Note: this package requires `openmp` for multithread processing, which is not supported on apple's `llvm`. If you're using macOS, you may install brew's `llvm` and set the corresponding reference before installing this package:
```bash
brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
```

A [docker](https://www.docker.com/) container of jupyter notebook can be constructed from the `Dockerfile`.

## Usage
### Import the package
To use the SimultaneousPLN package, import it as follows:
```python
import SimultaneousPLN as SimPLN
```
### Prepare the Data

Prepare your count data, offset, and covariates for different groups as lists of 2D NumPy arrays, where:
- **Rows** represent samples.
- **Columns** represent features.

Ensure that the number of columns (features) is the same across all arrays within the lists. The count data and offset for each group must have the exact same dimensions.

An example pipeline for generating simulation data is provided in the package:
```python
# generating adjacency matrices
As = SimPLN.generate_graph("ErdosRenyi", "ErdosRenyi", nodes=80, groups=30, common_p=0.1, p=0.8)
# generating precision matrix element values
values = [SimPLN.generate_value_uniform(nodes=80) for _ in range(30)]
# construct sparse precision matrices
Omegas = [SimPLN.sparse_Omega(As[i],values[i]) for i in range(30)]
# generate count data from PLN model
y = [SimPLN.generate_PLN(Omegas[i], nsample=200) for i in range(30)]
```

### Model Construction

You can construct the model as follows:

```python
model = SimPLN.SPLN(y, Offset, z)
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
For more detailed usage and examples, please refer to example folder included in the package.

# References

--- 
>Changhao Ge, Hongzhe Li, 2024. _Simultaneous estimation of many sparse networks via hierarchical poisson log-normal model._ [arxiv.org/abs/2409.12275](https://arxiv.org/abs/2409.12275)
