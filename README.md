<!--
SPDX-FileCopyrightText: 2024 2024 Quantum Application Lab

SPDX-License-Identifier: Apache-2.0
-->

# qreorder

**qreorder** is a specialized tool for reordering matrices using quantum annealing to maintain sparsity after LU decomposition, known as fill-in minimization. This approach is essential for quantum and classical computing applications where efficient matrix factorization is critical. The implemented approach solves multiple QUBO's for the equivalent chordal completion problem while iteratively increasing constraints using 'Bender's Cuts'. 

![License](https://img.shields.io/badge/license-apache-2)

---

## Installation

To install and set up **qreorder**, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/QuantumApplicationLab/qreorder.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd qreorder
   ```

3. **Install dependencies**:
   Ensure you have Python 3.8+ and `pip` installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The reordering consists of 4 steps:

1. Several classical preprocessing steps contained in qreorder/preprocessing.py. Including the removal of graph leaves, shrinking loops and reducing cliques.
2. Methods for creating the QUBO formulation for solving the chordal completion problem in qreorder/quantum/qubo.py.
3. The BenderQUBO class in qreorder/quantum/bender.py for organizing an optimization routine that iteratively adds constraints to a QUBO formulation.
4. The quantum solver in qreorder/quantum/quantum_solver.py is the class that combines all the above and uses methods to output an optimized reorder for an input matrix.

## Features

- **Quantum Annealing for Sparsity**: Uses quantum annealing techniques to optimize matrix reordering for LU decomposition, ensuring minimal fill-in.
- **Matrix File Support**: Supports input/output with `.csv` files for easy integration with other data processing tools.


## Example: Basic Matrix Reordering

Here’s a simple example of reordering a matrix to maintain sparsity post LU decomposition. The output is a list that denotes the matrix reorder where output[i]=j implies that the i'th row and column of the input matrix are reordered to be the j'th column for the new matrix.

```python
#import solver
from qreorder.quantum.quantum_solver import QuantumSolver
import numpy as np

# Example matrix to reorder
matrix = np.array([
    [1, 0, 2],
    [0, 3, 0],
    [4, 0, 5]
])

# Call the solve function from the solver module
solver = QuantumSolver()
matrix_reorder = solver.get_ordering(matrix)
print(matrix_reorder)
```


## Contributing

We welcome contributions! Follow these steps to get started:

1. **Fork** this repository.
2. **Create a branch** (`git checkout -b feature-name`).
3. **Commit** your changes (`git commit -m 'Add feature XYZ'`).
4. **Push** to the branch (`git push origin feature-name`).
5. **Open a Pull Request**.

Please make sure to update tests as appropriate and adhere to coding guidelines.

## License

This project is licensed under the apache-2.0 license - see the [LICENSE](LICENSE) file for details.

