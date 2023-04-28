use ndarray::prelude::*;

pub fn build_qr(
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

pub fn apply_givenss_rotation(component_j: f64, component_i: f64) -> (f64, f64) {
    let j_is_larger: bool = component_j.abs() > component_i.abs();
    let largest_component: f64 = if j_is_larger {
        component_j
    } else {
        component_i
    };

    let numerator: f64 = largest_component.signum();

    get_givens_components(component_j, component_i, numerator, j_is_larger)
}
