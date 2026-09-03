import sys
import numpy as np
from main import (
    column_by_column_RREF,
    clear_console,
)

from fractional_version import (
    convert_ndarray_to_fraction_ndarray,
    get_augmented_matrix_csv,
    get_yes_no_answer,
    get_csv_file_path,
    get_value,
)

def get_augmented_matrix_input(use_fractions: bool = False) -> np.ndarray:
    n_rows = get_value("How many rows are in your matrix? (integer) ", mode="int")
    if n_rows <= 0:
        raise ValueError(f"There has to be at least 1 row! Current amount: {n_rows}")

    matrix = []

    for row_i in range(n_rows):
        matrix_row = []

        mode = "fraction" if use_fractions else "float"

        for unknown_j in range(n_rows):
            unknown_ij = get_value(
                f"What is the value in position {unknown_j + 1} in row {row_i + 1}? ({mode}) ",
                mode=mode,
            )
            matrix_row.append(unknown_ij)

        matrix.append(matrix_row)

    if use_fractions:
        return np.array(matrix, dtype=object)
    return np.array(matrix, dtype=np.float64)

def get_augmented_matrix() -> tuple[np.ndarray, bool]:
    while True:
        clear_console()

        input_option = input("Do you want to input the matrix manually (M) or use a .csv file (F)? ").strip().lower()

        if input_option not in ["m", "f"]:
            print(f"The option '{input_option}' is invalid")
            input("Press enter to try again")
            continue

        break
        
    use_fractions = get_yes_no_answer(
        "Do you want to use fractions?" \
        "\nNote 1: It makes the calculations slower, but far more precise" \
        "\nNote 2: It won't force all the numbers to be a fractions, it'll automatically convert numbers to a fraction if there's a decimal place or '/' in the data" \
        "\nAnswer (y/n): "
    )
    force_fractions = False
    show_as_fractions = False
    if use_fractions:
        if input_option == "f":
            force_fractions = get_yes_no_answer(
                "Do you want to force fractions?" \
                "\nNote: This WILL force all the numbers to be fractions no matter what. This will make all future calculations immune to floating point precision errors" \
                "\n      But it'll slow down the calculations" \
                "\nAnswer (y/n): "
            )

        show_as_fractions = get_yes_no_answer(
            "Do you want to show the RREF matrix in fractional form?" \
            "\nAnswer (y/n): "
        )

    if input_option == "f":
        return get_augmented_matrix_csv(get_csv_file_path(), use_fractions=use_fractions, force_fractions=force_fractions), show_as_fractions
    return get_augmented_matrix_input(use_fractions=use_fractions), show_as_fractions

if __name__ == "__main__":
    # RREF_func = gauss_jordan_elimination
    RREF_func = column_by_column_RREF

    og_matrix, show_as_fraction = get_augmented_matrix()
    # augmented_matrix = get_augmented_matrix_input(use_fractions=True)
    # augmented_matrix = get_augmented_matrix_csv("equation_1.csv", force_fractions=True)
    # augmented_matrix = get_augmented_matrix_csv("equation_2.csv")
    # augmented_matrix = get_augmented_matrix_csv("fraction_equation_1.csv")
    
    clear_console()
    print("Original matrix:")
    # print(og_matrix)
    print(og_matrix.astype(str))
    print()
    print("Inverse matrix:")
    if og_matrix.shape[0] != og_matrix.shape[1]:
        print("ERROR: The matrix must be a square matrix")
        sys.exit(-1)
        
    if abs(np.linalg.det(og_matrix.astype(float))) <= 1e-10:
        print("This matrix does not have an inverse")
        sys.exit(0)
        
    if og_matrix.dtype == object:
        augmented_matrix = np.concatenate([og_matrix, convert_ndarray_to_fraction_ndarray(np.identity(og_matrix.shape[0]))], axis=1)
    else:
        augmented_matrix = np.concatenate([og_matrix, np.identity(og_matrix.shape[0])], axis=1)
        
    if show_as_fraction:
        print(RREF_func(augmented_matrix)[:, og_matrix.shape[0]:].astype(str))
    else:
        print(RREF_func(augmented_matrix)[:, og_matrix.shape[0]:].astype(float))