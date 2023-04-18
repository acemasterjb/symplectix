use ndarray::prelude::*;

fn build_qr(
    mut component_r: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
    row: usize,
    component_c: f64,
    component_s: f64,
) -> (
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
) {
    let mut component_q: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        Array::eye(component_r.len_of(Axis(0)));

    component_q[[row, row]] = component_c;
    component_q[[row - 1, row - 1]] = component_c;
    component_q[[row, row - 1]] = component_s;
    component_q[[row - 1, row]] = -component_s;

    component_r = component_q.t().dot(&component_r);

    (component_q, component_r)
}

fn get_givens_components(
    component_j: f64,
    component_i: f64,
    numerator: f64,
    j_is_larger: bool,
) -> (f64, f64) {
    let component_t: f64;
    let component_s: f64;
    let component_c: f64;

    if j_is_larger {
        component_t = component_i / component_j;
        component_s = numerator / (1.0 + component_t.powi(2)).sqrt();
        component_c = component_s * component_t;
    } else {
        component_t = component_j / component_i;
        component_c = numerator / (1.0 + component_t.powi(2)).sqrt();
        component_s = component_c * component_t;
    }

    (component_c, component_s)
}

fn apply_givenss_rotation(component_j: f64, component_i: f64) -> (f64, f64) {
    let j_is_larger: bool = component_j.abs() > component_i.abs();
    let largest_component: f64 = if j_is_larger {
        component_j
    } else {
        component_i
    };

    let numerator: f64 = largest_component.signum();

    get_givens_components(component_j, component_i, numerator, j_is_larger)
}

fn main() {
    let example_matrix: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        array![[1., 2., 3.], [-1., 1., 1.], [1., 1., 1.], [1., 1., 1.]];
    let n_rows: usize = example_matrix.len_of(Axis(0));

    let mut q: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = Array::eye(n_rows);
    let mut r: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = example_matrix.clone();
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

    println!("Q:\n{q}\n\nR:\n{r}");
}
