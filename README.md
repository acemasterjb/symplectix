# Implementation of the Symplectic Method

## QR Algorithm
This program uses the [Givens Rotation](https://en.wikipedia.org/wiki/Givens_rotation) method for calculating the QR Decomposition of a given matrix `A` s.t.:

**A = QR**

after running `A` on a QR decomposition algorithm.

For now, only an example matrix from [James V. Lamber's class notes](https://www.math.usm.edu/lambers/mat610/class0208.pdf) is used to demonstrate this algorithm.

---
### Prerequisites
- Rust >= 1.67.1
### Usage Instructions (testing)
```bash
cargo run
```
### Build Instructions
```bash
cargo build -r
```
Then the compiled program can be run with
```bash
./target/release/symplectix
```
