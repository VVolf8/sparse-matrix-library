# Sparse Matrix Library

## Description

The Sparse Matrix Library is a highly optimized and modular library designed for efficient operations on sparse matrices. It provides support for multiple sparse matrix formats, including:

- **CSR (Compressed Sparse Row):** Optimized for fast row-based operations such as matrix-vector multiplication.
- **CSC (Compressed Sparse Column):** Ideal for fast column-based operations, including efficient column extraction.
- **COO (Coordinate Format):** Useful for easy matrix construction before conversion to optimized formats.

This library is particularly useful for machine learning, numerical computing, and scientific applications where large sparse matrices are frequently used.

## Features

- **Efficient sparse matrix operations:** Addition, multiplication, scaling, and scalar operations.
- **Support for multiple formats:** Conversion between CSR, CSC, and COO.
- **Matrix factorization methods:** LU decomposition, Cholesky decomposition.
- **Iterative solvers:** Conjugate Gradient Method for solving linear systems.
- **Reordering algorithms:** Approximate Minimum Degree (AMD) ordering and Nested Dissection for fill-in minimization.
- **Row and column operations:** Extraction and replacement of rows/columns with optimized format selection.

## Installation

To install the library, use:

```sh
go get github.com/VVolf8/sparse-matrix-library
```

## Usage

Example of matrix creation and multiplication:

```go
package main

import (
    "fmt"
    "github.com/VVolf8/sparse-matrix-library"
)

func main() {
    A := sparse.NewSparseMatrix(2, 3, []float64{1, 2, 3}, []int{0, 2, 1}, []int{0, 2, 3})
    B := sparse.NewSparseMatrix(3, 2, []float64{3, 4, 5, 6}, []int{0, 1, 0, 1}, []int{0, 1, 2, 4})
    C := A.Multiply(B)
    C.Print()
}
```

## License

This project is licensed under a **custom license** based on Apache 2.0, with additional restrictions.

- **Free for personal and non-commercial use**.
- **Forbidden to use in paid or commercial projects without explicit permission from the author**.
- See the [`LICENSE`](./LICENSE) file for full details.

## Contribution

Contributions are welcome! Please follow the standard pull request workflow if you would like to contribute to this project.

## Contact

For any questions or issues, feel free to create an issue in the repository.

