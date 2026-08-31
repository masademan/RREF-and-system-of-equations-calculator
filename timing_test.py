import time
import json
import numpy as np
from fractional_version import convert_ndarray_to_fraction_ndarray
from main import gauss_jordan_elimination, column_by_column_RREF, clear_console


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        if kwargs.get("use_ns", False):
            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            end = time.perf_counter_ns()
        else:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
        return result, end - start

    return wrapper


@timer_decorator
def timed_gauss_jordan_elimination(matrix, use_ns=False):
    return gauss_jordan_elimination(matrix)


@timer_decorator
def timed_column_by_column_RREF(matrix, use_ns=False):
    return column_by_column_RREF(matrix)


def generate_integer_solution_matrix(
    n: int, max_element_val: int = 5, max_solution: int = 5, use_floats: bool = False, convert_to_fraction: bool = False, return_answer: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if use_floats:
        L = np.tril(np.random.uniform(-max_element_val, max_element_val + 1, size=(n, n)), -1)
        np.fill_diagonal(L, np.random.choice([-1, 1], size=n))

        U = np.triu(np.random.uniform(-max_element_val, max_element_val + 1, size=(n, n)), 1)
        np.fill_diagonal(U, np.random.choice([-1, 1], size=n))

        A = np.dot(L, U)

        solutions = np.random.uniform(-max_solution, max_solution + 1, size=(n, 1))

        b = np.dot(A, solutions)

        augmented_matrix = np.concatenate((A, b), axis=1).astype(np.float64)
    else:
        L = np.tril(np.random.randint(-max_element_val, max_element_val + 1, size=(n, n)), -1)
        np.fill_diagonal(L, np.random.choice([-1, 1], size=n))

        U = np.triu(np.random.randint(-max_element_val, max_element_val + 1, size=(n, n)), 1)
        np.fill_diagonal(U, np.random.choice([-1, 1], size=n))

        A = np.dot(L, U)

        # solutions = np.ones((n, 1), dtype=int)
        solutions = np.random.randint(-max_solution, max_solution + 1, size=(n, 1))

        b = np.dot(A, solutions)

        augmented_matrix = np.concatenate((A, b), axis=1).astype(np.float64)

    if convert_to_fraction: augmented_matrix = convert_ndarray_to_fraction_ndarray(augmented_matrix)

    if not return_answer:
        return augmented_matrix

    return augmented_matrix, solutions


# Google Sheets data:
# https://docs.google.com/spreadsheets/d/1qa8tAm3_GasE6jLSZKdni2R_my4WJndrSQPZkayF0sc/edit?usp=sharing
if __name__ == "__main__":
    clear_console()

    use_ns = False                                                      # Whether or not to time the functions using nanoseconds (larger number, but removes the issue of floating point precision errors)
    # sizes_to_test = [5, 10, 50, 100, 500]
    sizes_to_test = [5, 10, 50, 100, 200, 300]                                # Options for the size of the augmented matrix: (n, n + 1)
    # sizes_to_test = [5, 10, 50, 100, 500, 1000]
    num_trials = 5                                                      # Number of times to time the operation and then average the times
    functions = [gauss_jordan_elimination, column_by_column_RREF]       # Functions to time

    data = {}
    for func in functions:
        @timer_decorator
        def timed_func(matrix, use_ns=False):
            return func(matrix)

        func_data = {}

        for size in sizes_to_test:
            total_time = 0

            for _ in range(num_trials):
                total_time += timed_func(generate_integer_solution_matrix(size, max_element_val=5, max_solution=5, convert_to_fraction=False), use_ns=use_ns)[1]

            avg_time = total_time / num_trials

            func_data[size] = avg_time

            print(f"Function '{func.__name__}' finished size {size}")

        print(f"Function '{func.__name__}' finished testing")
        print()

        data[func.__name__] = func_data

    print(json.dumps(data, indent=4))
