import os
import csv
import numpy as np
from typing import Literal
from fractions import Fraction
from main import (
    augmented_matrix_to_equation_strs,
    systems_of_equation_solver,
    gauss_jordan_elimination,
    format_variable_values,
    column_by_column_RREF,
    get_csv_file_path,
    clear_console,
    is_float,
    is_int,
)

AUGMENTED_MATRIX_PATH = "augmented_matrices/"


def is_fraction(n: str) -> bool:
    try:
        _ = Fraction(n)
        return True
    except ValueError:
        return False
    except Exception as e:
        return RuntimeError(f"The input '{n}' gave an unexpected error: {e}")


def get_value(question: str, mode: Literal["int", "float", "fraction"]) -> int | float | Fraction:
    if mode not in ["int", "float", "fraction"]:
        return NotImplementedError(f"Mode '{mode}' has not yet been implemented")
    while True:
        clear_console()
        answer = input(question).strip()

        if mode == "int":
            if not is_int(answer):
                print(f"The response '{answer}' is not an integer")
                input("Press enter to try again")
                continue

            return int(answer)

        elif mode == "float":
            if not is_float(answer):
                print(f"The response '{answer}' is not a float")
                input("Press enter to try again")
                continue

            return float(answer)

        elif mode == "fraction":
            if not is_fraction(answer):
                print(f"The response '{answer}' is not a fraction")
                input("Press enter to try again")
                continue

            return Fraction(answer)


def get_augmented_matrix_input(use_fractions: bool = False) -> np.ndarray:
    n_equations = get_value("How many linear equations are there? (integer) ", mode="int")
    if n_equations <= 0:
        raise ValueError(f"There has to be at least 1 linear equation! Current amount: {n_equations}")

    n_unknowns = get_value("How many unknowns are there? (integer) ", mode="int")
    if n_unknowns <= 0:
        raise ValueError(f"There has to be at least 1 unknown! Current amount: {n_unknowns}")

    augmented_matrix = []

    for equation_i in range(n_equations):
        augmented_matrix_row = []

        mode = "fraction" if use_fractions else "float"

        for unknown_j in range(n_unknowns):
            unknown_ij = get_value(
                f"What is the coefficient of x_{unknown_j + 1} in equation {equation_i + 1}? ({mode}) ",
                mode=mode,
            )
            augmented_matrix_row.append(unknown_ij)

        augmented_matrix_row.append(
            get_value(
                f"What is the value of equation {equation_i + 1}? ({mode}) ",
                mode=mode,
            )
        )

        augmented_matrix.append(augmented_matrix_row)

    if use_fractions:
        return np.array(augmented_matrix, dtype=object)
    return np.array(augmented_matrix, dtype=np.float64)


def get_augmented_matrix_csv(file_path: str, use_fractions: bool = False, force_fractions: bool = False) -> np.ndarray:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    if not file_path.startswith(AUGMENTED_MATRIX_PATH):
        file_path = AUGMENTED_MATRIX_PATH + file_path

    if not os.path.exists(file_path):
        return FileNotFoundError(f"The file at '{file_path}' does not exist")

    with open(file_path, "r") as f:
        csv_str = f.read()
    use_fractions = (("/" in csv_str or "." in csv_str) and use_fractions) or force_fractions

    if not use_fractions:
        augmented_matrix = np.genfromtxt(file_path, delimiter=",", skip_header=1, filling_values=0, dtype=np.float64)
        return augmented_matrix

    else:
        matrix_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                parsed_row = []

                for val in row:
                    val_str = val.strip()
                    if not val_str:
                        parsed_row.append(Fraction(0))
                    else:
                        parsed_row.append(Fraction(val_str))

                matrix_data.append(parsed_row)

        augmented_matrix = np.array(matrix_data, dtype=object)
        return augmented_matrix


def write_augmented_matrix_csv(file_path: str, augmented_matrix: np.ndarray) -> None:
    header = []
    for i in range(augmented_matrix.shape[1] - 1):
        header.append(f"x_i{i + 1}")
    header.append("y")
    header = ",".join(header)

    if not file_path.endswith(".csv"):
        file_path += ".csv"
    if not file_path.startswith(AUGMENTED_MATRIX_PATH):
        file_path = AUGMENTED_MATRIX_PATH + file_path

    if augmented_matrix.dtype == object:
        np.savetxt(file_path, augmented_matrix, delimiter=",", header=header, comments="", fmt="%s")
    else:
        np.savetxt(file_path, augmented_matrix.astype(np.float64), delimiter=",", header=header, comments="")


def get_yes_no_answer(question: str) -> bool:
    while True:
        clear_console()

        answer = input(question).strip().lower()

        if answer not in ["y", "n"]:
            print(f"The answer '{answer}' is invalid")
            input("Press enter to try again")
            continue

        return answer == "y"


def get_augmented_matrix() -> np.ndarray:
    while True:
        clear_console()

        input_option = input("Do you want to input the matrix manually (M) or use a .csv file (F)? ").strip().lower()

        if input_option not in ["m", "f"]:
            print(f"The option '{input_option}' is invalid")
            input("Press enter to try again")
            continue

        break

    if input_option == "m":
        return get_augmented_matrix_input(
            use_fractions=get_yes_no_answer(
                "Do you want to use fractions?" \
                "\nNote: It makes the calculations slower, but far more precise" "\nAnswer (y/n): "
            )
        )
    use_fractions = get_yes_no_answer(
        "Do you want to use fractions?" \
        "\nNote 1: It makes the calculations slower, but far more precise" \
        "\nNote 2: It won't force all the numbers to be a fractions, it'll automatically convert numbers to a fraction if there's a decimal place or '/' in the data" \
        "\nAnswer (y/n): "
    )
    force_fractions = False
    if use_fractions:
        force_fractions = get_yes_no_answer(
            "Do you want to force fractions?" \
            "\nNote: This WILL force all the numbers to be fractions no matter what. This will make all future calculations immune to floating point precision errors" \
            "\n      But it'll slow down the calculations" \
            "\nAnswer (y/n): "
        )
    return get_augmented_matrix_csv(get_csv_file_path(), use_fractions=use_fractions, force_fractions=force_fractions)


def convert_list_to_fraction_list(num_list: list[int | float]) -> list[Fraction]:
    for i in range(len(num_list)):
        if type(num_list[i]).__name__ == "list":
            num_list[i] = convert_list_to_fraction_list(num_list[i])
        else:
            num_list[i] = Fraction(num_list[i]).limit_denominator()

    return num_list


def convert_ndarray_to_fraction_ndarray(matrix: np.ndarray) -> np.ndarray:
    to_list_matrix = matrix.tolist()

    return np.array(convert_list_to_fraction_list(to_list_matrix))


if __name__ == "__main__":
    RREF_func = gauss_jordan_elimination
    # RREF_func = column_by_column_RREF

    augmented_matrix = get_augmented_matrix()
    # augmented_matrix = get_augmented_matrix_input(use_fractions=True)
    # augmented_matrix = get_augmented_matrix_csv("equation_1.csv", force_fractions=True)
    # augmented_matrix = get_augmented_matrix_csv("equation_2.csv")
    # augmented_matrix = get_augmented_matrix_csv("fraction_equation_1.csv")
    clear_console()
    print("Original augmented matrix:")
    print(augmented_matrix.astype(str))
    print()
    print("Equations:")
    print(augmented_matrix_to_equation_strs(augmented_matrix, int_if_possible=True, all_plus_form=True, all_vars=False, show_all_coefficients=True))
    print()
    print("RREF:")
    print(RREF_func(augmented_matrix).astype(str))
    print()
    print("Variable values:")
    print(
        format_variable_values(
            systems_of_equation_solver(
                augmented_matrix, all_str=True, int_if_possible=True, all_plus_form=True, show_all_coefficients=True, RREF_func=RREF_func
            )
        )
    )
    print()
