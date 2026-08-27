import os
import subprocess
import numpy as np
from typing import Literal, Callable

AUGMENTED_MATRIX_PATH = "augmented_matrices/"


def clear_console():
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def is_float(n: str) -> bool:
    try:
        _ = float(n)
        return True
    except ValueError:
        return False
    except Exception as e:
        return RuntimeError(f"The input '{n}' gave an unexpected error: {e}")


def is_int(n: str) -> bool:
    try:
        _ = int(n)
        return True
    except ValueError:
        return False
    except Exception as e:
        return RuntimeError(f"The input '{n}' gave an unexpected error: {e}")


def get_value(question: str, mode: Literal["int", "float"]) -> int | float:
    if mode not in ["int", "float"]:
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


def get_augmented_matrix_input() -> np.ndarray:
    n_equations = get_value("How many linear equations are there? (integer) ", mode="int")
    if n_equations <= 0:
        raise ValueError(f"There has to be at least 1 linear equation! Current amount: {n_equations}")

    n_unknowns = get_value("How many unknowns are there? (integer) ", mode="int")
    if n_unknowns <= 0:
        raise ValueError(f"There has to be at least 1 unknown! Current amount: {n_unknowns}")

    augmented_matrix = []

    for equation_i in range(n_equations):
        augmented_matrix_row = []

        for unknown_j in range(n_unknowns):
            unknown_ij = get_value(
                f"What is the coefficient of x_{unknown_j + 1} in equation {equation_i + 1}? (float) ",
                mode="int",
            )
            augmented_matrix_row.append(unknown_ij)

        augmented_matrix_row.append(
            get_value(
                f"What is the value of equation {equation_i + 1}? (float) ",
                mode="float",
            )
        )

        augmented_matrix.append(augmented_matrix_row)

    return np.array(augmented_matrix, dtype=np.float64)


def get_csv_file_path() -> str:
    files = [f for f in os.listdir("./augmented_matrices/")]

    while True:
        clear_console()

        for i, file in enumerate(files):
            print(f"{i + 1}: {file}")
        print()

        answer = input("Which file do you want? ")
        if not is_int(answer):
            print(f"'{answer}' is an invalid choice, input an integer")
            input("Press continue to try again")
            continue

        answer = int(answer)
        if answer > len(files):
            print(f"That choice was too big (choice has to be <= {len(files)})")
            input("Press enter to try again")
            continue
        if answer < 1:
            print("That choice was too small (choice has to be >= 1)")
            input("Press enter to try again")
            continue
        break

    return files[answer - 1]


def get_augmented_matrix_csv(file_path: str) -> np.ndarray:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    if not file_path.startswith(AUGMENTED_MATRIX_PATH):
        file_path = AUGMENTED_MATRIX_PATH + file_path

    if not os.path.exists(file_path):
        return FileNotFoundError(f"The file at '{file_path}' does not exist")

    augmented_matrix = np.genfromtxt(file_path, delimiter=",", skip_header=1, filling_values=0, dtype=np.float64)
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

    if not os.path.exists(file_path):
        return FileNotFoundError(f"The file at '{file_path}' does not exist")

    np.savetxt(file_path, augmented_matrix, delimiter=",", header=header, comments="")


def augmented_matrix_to_equation_strs(
    augmented_matrix: np.ndarray, int_if_possible: bool = True, all_plus_form: bool = False, all_vars: bool = True, show_all_coefficients: bool = True
) -> str:
    if not is_valid_augmented_matrix(augmented_matrix):
        raise ValueError("The matrix given is not a valid augmented matrix")

    all_equations = []

    for row in augmented_matrix:
        single_equation = []

        for coeff_idx in range(row.shape[0] - 1):
            if row[coeff_idx] == 0 and not all_vars:
                continue

            coeff = row[coeff_idx]
            if int(coeff) == coeff and int_if_possible:
                coeff = int(coeff)

            if all_plus_form:
                if coeff != 1 or show_all_coefficients:
                    single_equation.append(f"({coeff})x_{coeff_idx + 1}")
                else:
                    single_equation.append(f"x_{coeff_idx + 1}")
                single_equation.append("+")
            else:
                if len(single_equation) == 0:
                    if coeff != 1 or show_all_coefficients:
                        single_equation.append(f"({coeff})x_{coeff_idx + 1}")
                    else:
                        single_equation.append(f"x_{coeff_idx + 1}")
                else:
                    single_equation.append("+" if coeff > 0 else "-")
                    if abs(coeff) != 1 or show_all_coefficients:
                        single_equation.append(f"({abs(coeff)})x_{coeff_idx + 1}")
                    else:
                        single_equation.append(f"x_{coeff_idx + 1}")

        if all_plus_form and len(single_equation) > 0:
            single_equation.pop()

        if len(single_equation) == 0:
            if not show_all_coefficients:
                continue
            single_equation.append("0")

        single_equation.append("=")

        constant = row[-1]
        if int(constant) == constant and int_if_possible:
            constant = int(constant)

        single_equation.append(str(constant))

        all_equations.append(" ".join(single_equation))

    return "\n".join(all_equations)


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
        return get_augmented_matrix_input()
    return get_augmented_matrix_csv(get_csv_file_path())


def is_REF(matrix: np.ndarray) -> bool:
    # All zero rows are at the bottom of the matrix
    is_all_zero = (matrix == 0).all(axis=1)
    found_zero = False
    for item in is_all_zero:
        found_zero = found_zero or item
        if not item and found_zero:
            return False

    # If a row isn't all 0's, the first nonzero num (leading entry) has to be 1
    # In 2 vertically consecutive rows that aren't all 0's, the leading
    #  entry in the lower row appears farther to the right than the leading
    #  entry in the higher row.
    last_leading_entry_j = 0
    for i in range(matrix.shape[0]):
        row = matrix[i]

        if (row == 0).all():
            break

        for j in range(row.shape[0]):
            item = row[j]
            if item == 0:
                continue
            elif item == 1:
                if i != 0 and last_leading_entry_j >= j:
                    return False

                last_leading_entry_j = j
                break
            else:
                return False

    return True


def is_RREF(matrix: np.ndarray) -> bool:
    # Is a REF matrix
    if not is_REF(matrix):
        return False

    # Each column containing a leading entry has 0's everywhere else
    for i in range(matrix.shape[0]):
        row = matrix[i]

        if (row == 0).all():
            break

        for j in range(row.shape[0]):
            item = row[j]
            if item == 0:
                continue
            elif item == 1:
                if (matrix[:, j] == 0).sum() == 1:
                    return False
                break

    return True


def swap_rows(matrix: np.ndarray, a: int, b: int) -> np.ndarray:
    temp = matrix[a].copy()
    matrix[a] = matrix[b]
    matrix[b] = temp
    return matrix


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    epsilon = 1e-10

    # Make tiny decimals 0
    idxs = np.where(np.abs(matrix) <= epsilon)
    matrix[idxs] = 0

    # Round floats that are incredibly close to being an integer
    idxs = np.where(np.abs(matrix - np.floor(matrix)) <= epsilon)
    matrix[idxs] = np.floor(matrix[idxs])

    idxs = np.where(np.abs(matrix - np.ceil(matrix)) <= epsilon)
    matrix[idxs] = np.ceil(matrix[idxs])

    return matrix


def gauss_elimination(matrix: np.ndarray) -> np.ndarray:  # Turns a matrix into an REF matrix
    num_rows_to_ignore = 0

    while not is_REF(matrix):
        viewable_matrix = matrix[num_rows_to_ignore:]

        if (viewable_matrix == 0).all():
            break

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Find the leftmost column that isn't all 0's
        column_idx = 0
        while column_idx < viewable_matrix.shape[1]:
            if not (viewable_matrix[:, column_idx] == 0).all():
                break
            column_idx += 1

        # If the top row in the column found is a 0, swap it with another row
        if viewable_matrix[0, column_idx] == 0:
            row_idx = 0

            while row_idx < viewable_matrix.shape[0]:
                if viewable_matrix[row_idx, column_idx] != 0:
                    break

                row_idx += 1

            swap_rows(viewable_matrix, 0, row_idx)

        # If the entry in that top row is a, multiply the entire row by 1/a
        if viewable_matrix[0, column_idx] != 1:
            viewable_matrix[0, :] *= 1 / viewable_matrix[0, column_idx]

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Add suitable multiples of the top row to the rows below so that the entries below the 1 become 0
        viewable_matrix[1:] += np.outer(-viewable_matrix[1:, column_idx], viewable_matrix[0, :])

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Repeat, but ignore the top row
        num_rows_to_ignore += 1

    return matrix


def gauss_jordan_elimination(matrix: np.ndarray) -> np.ndarray:  # Turns a matrix into an RREF matrix
    gauss_elimination(matrix)

    # Starting at the last nonzero row and going upwards, add suitable multiples
    #  of each row to the rows above to introduce zeros above the leading entries
    row_idx = matrix.shape[0] - 1

    while row_idx >= 0:
        if (matrix[row_idx, :] == 0).all():
            row_idx -= 1
            continue

        column_idx = np.where(matrix[row_idx, :] == 1)[0][0]

        matrix[:row_idx] += np.outer(-matrix[:row_idx, column_idx], matrix[row_idx, :])

        if matrix.dtype != object:
            clean_matrix(matrix)

        row_idx -= 1

    return matrix


def column_by_column_RREF(matrix: np.ndarray) -> np.ndarray:  # Turns a matrix into an RREF matrix without making a substep of an REF matrix
    row_idx = 0

    while not is_RREF(matrix) and row_idx < matrix.shape[0]:
        if (matrix == 0).all():
            break

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Find the leftmost column that isn't all 0's
        column_idx = 0
        while column_idx < matrix.shape[1]:
            if not (matrix[row_idx:, column_idx] == 0).all():
                break
            column_idx += 1

        # If the top row in the column found is a 0, swap it with another row
        if matrix[row_idx, column_idx] == 0:
            sub_row_idx = 0

            while sub_row_idx < matrix.shape[0]:
                if matrix[sub_row_idx, column_idx] != 0:
                    break

                sub_row_idx += 1

            swap_rows(matrix, row_idx, sub_row_idx)

        # If the entry in that top row is a, multiply the entire row by 1/a
        if matrix[row_idx, column_idx] != 1:
            matrix[row_idx, :] *= 1 / matrix[row_idx, column_idx]

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Add suitable multiples of the top row to the rows below so that the entries below the 1 become 0
        matrix[row_idx + 1 :] += np.outer(-matrix[row_idx + 1 :, column_idx], matrix[row_idx, :])

        if matrix.dtype != object:
            clean_matrix(matrix)

        # Add suitable multiples of the top row to the rows above so that the entries below the 1 become 0
        if row_idx > 0:
            matrix[: row_idx - 1] += np.outer(-matrix[: row_idx - 1, column_idx], matrix[row_idx, :])

            if matrix.dtype != object:
                clean_matrix(matrix)

        # Repeat, but ignore the top row
        row_idx += 1

    return matrix


def is_valid_augmented_matrix(matrix: np.ndarray) -> bool:
    coefficients = matrix[:, :-1]
    constants = matrix[:, -1]

    if (coefficients == 0).all():
        return False

    if ((coefficients == 0).all(axis=1) & ~(constants == 0)).any():
        return False

    return True


def systems_of_equation_solver(
    augmented_matrix: np.ndarray,
    all_str: bool = True,
    int_if_possible: bool = True,
    all_plus_form: bool = False,
    show_all_coefficients: bool = True,
    RREF_func: Callable[[np.ndarray], np.ndarray] = gauss_jordan_elimination,
) -> list[str | float | int]:
    if not is_valid_augmented_matrix(augmented_matrix):
        raise ValueError("The matrix given is not a valid augmented matrix")

    RREF_matrix = RREF_func(augmented_matrix)
    # RREF_matrix = gauss_jordan_elimination(augmented_matrix)
    # RREF_matrix = column_by_column_RREF(augmented_matrix)
    variable_values = []

    if not is_valid_augmented_matrix(augmented_matrix):
        raise ValueError("The matrix given is not a valid augmented matrix")

    try:
        for row_idx in range(RREF_matrix.shape[0]):
            if not is_valid_augmented_matrix(RREF_matrix[row_idx:]):
                break

            for var_i in range(len(variable_values), np.where(RREF_matrix[row_idx] == 1)[0][0]):
                variable_values.append(f"x_{var_i + 1}")

            value = RREF_matrix[row_idx, -1]
            if int(value) == value and int_if_possible:
                value = int(value)

            if (RREF_matrix[row_idx, :-1] == 0).sum() == 1:
                if all_str:
                    variable_values.append(str(value))
                else:
                    variable_values.append(value)
            else:
                # Show the values as a string because it relies on other variables
                # Use "np.where(row != 0)[0][1:]" to get the coefficients of the other variables that are relied on
                str_pieces = [str(value)]

                variable_idxs = np.where(RREF_matrix[row_idx, :-1] != 0)[0][1:]
                variable_coefficients = -RREF_matrix[row_idx, variable_idxs]

                for variable_idx, variable_coefficient in zip(variable_idxs, variable_coefficients):
                    if all_plus_form:
                        str_pieces.append("+")

                        var_coeff = variable_coefficient
                        if int(var_coeff) == var_coeff and int_if_possible:
                            var_coeff = int(var_coeff)

                        if var_coeff != 1 or show_all_coefficients:
                            str_pieces.append(f"({var_coeff})x_{variable_idx + 1}")
                        else:
                            str_pieces.append(f"x_{variable_idx + 1}")
                    else:
                        var_coeff = variable_coefficient
                        if int(var_coeff) == var_coeff and int_if_possible:
                            var_coeff = int(var_coeff)

                        str_pieces.append("+" if var_coeff > 0 else "-")

                        if abs(var_coeff) != 1 or show_all_coefficients:
                            str_pieces.append(f"({abs(var_coeff)})x_{variable_idx + 1}")
                        else:
                            str_pieces.append(f"x_{variable_idx + 1}")

                variable_values.append(" ".join(str_pieces))

        for var_i in range(len(variable_values), RREF_matrix.shape[1] - 1):
            variable_values.append(f"x_{var_i + 1}")

        return variable_values
    except IndexError:
        raise ValueError("The matrix given is not a valid augmented matrix") from None
    except Exception as e:
        raise e


def format_variable_values(variable_values: list[str | float | int]) -> str:
    formatted_equations = []

    for var_i in range(len(variable_values)):
        formatted_equations.append(f"x_{var_i + 1} = " + str(variable_values[var_i]))

    return "\n".join(formatted_equations)


if __name__ == "__main__":
    """
    Show that the Ti-84 evo also has the option to show the RREF matrix after solving a system of equations
    """

    RREF_func = gauss_jordan_elimination
    # RREF_func = column_by_column_RREF

    # Customizable input for the augmented matrix
    augmented_matrix = get_augmented_matrix()
    # augmented_matrix = get_augmented_matrix_input()
    # augmented_matrix = get_augmented_matrix_csv("test.csv")
    # augmented_matrix = get_augmented_matrix_csv("equation_1.csv")
    # augmented_matrix = get_augmented_matrix_csv("equation_2.csv")
    clear_console()
    print("Original augmented matrix:")
    print(augmented_matrix)
    print()
    print("Equations:")
    print(augmented_matrix_to_equation_strs(augmented_matrix, int_if_possible=True, all_plus_form=True, all_vars=True, show_all_coefficients=True))
    print()
    print("RREF:")
    print(RREF_func(augmented_matrix))
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

    # Reading the matrix from a .txt, printing the equations, and then solving
    # matrix = get_augmented_matrix_csv("test.csv")
    # # print(matrix)
    # print(augmented_matrix_to_equation_strs(matrix, int_if_possible=True, all_plus_form=False, all_vars=True))
    # # print(gauss_elimination(matrix))
    # print()
    # # print(gauss_jordan_elimination(matrix))
    # # print("\n".join(systems_of_equation_solver(matrix, all_str=True, int_if_possible=True, all_plus_form=True)))
    # print(format_variable_values(systems_of_equation_solver(matrix, all_str=True, int_if_possible=False, all_plus_form=True)))

    # Making an augmented matrix and then solving the linear system
    # matrix = np.array(
    #     [
    #         [0, 0, -2, 0, 7, 12],
    #         [2, 4, -10, 6, 12, 28],
    #         [2, 4, -5, 6, -5, -1],
    #     ],
    #     dtype=np.float64,
    # )

    # # print(gauss_elimination(matrix))
    # # print()
    # print(gauss_jordan_elimination(matrix))
    # print()
    # # print("\n".join(systems_of_equation_solver(matrix, all_str=True, int_if_possible=True, all_plus_form=False)))
    # print(format_variable_values(systems_of_equation_solver(matrix, all_str=True, int_if_possible=True, all_plus_form=False)))

    # Manually input the coefficients of the variables and the constant, print the equations, and then solve
    # augmented_matrix = get_augmented_matrix_input()
    # clear_console()
    # print("Equations:")
    # print(augmented_matrix_to_equation_strs(augmented_matrix, int_if_possible=True, all_plus_form=False, all_vars=True))
    # print()
    # print("Variable values:")
    # print(format_variable_values(systems_of_equation_solver(augmented_matrix, all_str=True, int_if_possible=True, all_plus_form=False)))
