from math import copysign, sqrt
from typing import Tuple

import numpy as np


def build_QR(
    component_R: np.ndarray,
    row: int,
    component_c: float,
    component_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    component_Q = np.identity(component_R.shape[0])

    component_Q[row][row] = component_c
    component_Q[row - 1][row - 1] = component_c
    component_Q[row][row - 1] = component_s
    component_Q[row - 1][row] = -component_s

    component_R = component_Q.transpose().dot(component_R)

    return component_Q, component_R


def get_givens_components(
    component_j: float, component_i: float, numerator: float, j_is_larger: bool
) -> Tuple[float, float]:
    component_t: float = 0.0
    component_s: float = 0.0
    component_c: float = 0.0

    if j_is_larger:
        component_t = component_i / component_j
        component_s = numerator / sqrt(1 + component_t**2)
        component_c = component_s * component_t
    else:
        component_t = component_j / component_i
        component_c = numerator / sqrt(1 + component_t**2)
        component_s = component_c * component_t

    return component_c, component_s


def apply_givens_rotation(
    component_j: float, component_i: float
) -> Tuple[float, float]:
    j_is_larger = abs(component_j) > abs(component_i)
    largest_component: float = component_j if j_is_larger else component_i

    numerator: float = 0.0
    if abs(largest_component) > 0:
        numerator = copysign(1.0, largest_component)

    return get_givens_components(component_j, component_i, numerator, j_is_larger)


def main():
    example_matrix: np.ndarray = np.array(
        [
            [1, 2, 3],
            [-1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )

    Q = np.identity(example_matrix.shape[0])
    R = example_matrix.copy()
    del example_matrix

    row_padding = 0
    for column in range(R.shape[1]):
        for row in range(R.shape[0] - 1, row_padding, -1):
            a = R[row - 1][column]
            b = R[row][column]
            c, s = apply_givens_rotation(b, a)

            Q_mn, R = build_QR(R, row, c, s)
            Q = Q.dot(Q_mn)
        row_padding += 1

    print(f"Q:\n{Q}\n\nR:\n{R}")


if __name__ == "__main__":
    main()
