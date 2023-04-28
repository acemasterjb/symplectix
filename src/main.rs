use ndarray::prelude::*;
use ndarray_linalg::Determinant;

mod formulae;
use crate::formulae::*;

fn find_qr_decomposition_of(
    input_matrix: &ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> (
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
) {
    let n_rows: usize = input_matrix.len_of(Axis(0));

    let mut r: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = input_matrix.clone();
    let mut q: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = Array::eye(n_rows);
    let mut row_padding: usize = 1;

    for column in 0..r.len_of(Axis(1)) {
        for row in (row_padding..n_rows).rev() {
            let a: f64 = r[[row - 1, column]];
            let b: f64 = r[[row, column]];
            let (c, s) = apply_givenss_rotation(b, a);

            let q_mn: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;
            (q_mn, r) = build_qr(r, row, c, s);
            q = q.dot(&q_mn);
        }
        row_padding += 1;
    }

    (q, r)
}

fn main() {
    let example_matrix: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        array![[2., 2., 3.], [4., 5., 6.], [7., 8., 9.]];

    let q: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;
    let r: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>;

    (r, q) = find_qr_decomposition_of(&example_matrix);

    let r_det: f64 = r.det().unwrap_or_default();
    let example_matrix_det: f64 = example_matrix.det().unwrap_or_default();

    println!("Q:\n{q}\n\nR:\n{r}");
    println!("\n\ndet(R): {r_det}");
    println!("det(A): {example_matrix_det}");
}

#[cfg(test)]
mod tests {
    use ndarray_linalg::aclose;
    use ndarray_linalg::random_hermite;
    use ndarray_linalg::Determinant;

    use crate::find_qr_decomposition_of;

    #[test]
    fn check_validity() {
        for size in 5..50 {
            let test_case: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
                random_hermite(size);

            let (_, r) = find_qr_decomposition_of(&test_case);

            aclose(
                r.det().unwrap_or_default(),
                test_case.det().unwrap_or_default(),
                1e-5,
            )
        }
    }
}
