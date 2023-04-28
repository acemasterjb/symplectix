# Implementation of the Symplectic Method

## QR Algorithm
This program uses the [Givens Rotation](https://en.wikipedia.org/wiki/Givens_rotation) method for calculating the QR Decomposition of a given matrix `A` s.t.:

**A = QR**

after running `A` on a QR decomposition algorithm.

---
### Prerequisites
- Rust >= 1.67.1
### Testing
These tests test the validity of the QR decomposition algorithm.

Currently, it checks if the determinant of the input is [close](https://docs.rs/ndarray-linalg/0.16.0/ndarray_linalg/assert/fn.aclose.html) to the determinant of its R component.

[Here's why](https://en.wikipedia.org/wiki/QR_decomposition#Connection_to_a_determinant_or_a_product_of_eigenvalues).
```bash
cargo test
```
### Usage Instructions (no build)
This will run
```bash
cargo run
```
### Usage Instructions (w/ building)
```bash
cargo build -r
```
Then the compiled program can be run with
```bash
./target/release/symplectix
```

