import argparse
import csv
import shutil
from pathlib import Path


def parse_matrix_token(token, matrix_size, modulus):
    """
    Parse one compact matrix token.

    For matrix_size = 3 and modulus = 2:
        100010001 -> 3x3 identity matrix

    Row-major order:
        11, 12, 13,
        21, 22, 23,
        31, 32, 33
    """
    token = token.strip()

    if not token:
        raise ValueError("Empty matrix token.")

    if "," in token:
        parts = token.split(",")
    else:
        if modulus > 10:
            raise ValueError(
                "For modulus > 10, use comma-separated matrix entries."
            )
        parts = list(token)

    try:
        entries = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid matrix token: {token}") from error

    expected_length = matrix_size * matrix_size

    if len(entries) != expected_length:
        raise ValueError(
            f"Matrix token must contain {expected_length} entries, got "
            f"{len(entries)}: {token}"
        )

    for value in entries:
        if value < 0 or value >= modulus:
            raise ValueError(
                f"Entry {value} is outside F_{modulus}: {token}"
            )

    return tuple(value % modulus for value in entries)


def format_matrix_token(matrix, modulus):
    """
    Convert a matrix tuple back to compact text.
    """
    separator = "" if modulus <= 10 else ","
    return separator.join(str(value % modulus) for value in matrix)


def transpose_matrix(matrix, matrix_size):
    """
    Transpose a square matrix stored in row-major order.
    """
    n = matrix_size

    return tuple(
        matrix[column * n + row]
        for row in range(n)
        for column in range(n)
    )


def entry(matrix, matrix_size, row, column):
    """
    Return matrix[row, column] from a row-major flat tuple.
    """
    return matrix[row * matrix_size + column]


def read_scheme(filename, matrix_size, modulus, transpose_c=False):
    """
    Read one compact-token scheme file.

    Each nonblank, noncomment line must contain:

        A_token B_token C_token
    """
    scheme = []

    with open(filename, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_number} must contain exactly three "
                    f"matrix tokens: {line}"
                )

            A = parse_matrix_token(parts[0], matrix_size, modulus)
            B = parse_matrix_token(parts[1], matrix_size, modulus)
            C = parse_matrix_token(parts[2], matrix_size, modulus)

            if transpose_c:
                C = transpose_matrix(C, matrix_size)

            scheme.append((A, B, C))

    if not scheme:
        raise ValueError(f"No scheme rows found in file: {filename}")

    return tuple(scheme)


def zero_factor_rows(scheme):
    """
    Return 1-based row numbers where A, B, or C is the zero matrix.
    """
    bad_rows = []

    for row_number, (A, B, C) in enumerate(scheme, start=1):
        if all(value == 0 for value in A):
            bad_rows.append(f"{row_number}:A")
        if all(value == 0 for value in B):
            bad_rows.append(f"{row_number}:B")
        if all(value == 0 for value in C):
            bad_rows.append(f"{row_number}:C")

    return bad_rows


def brent_errors(scheme, matrix_size, modulus, max_errors=None):
    """
    Check Brent equations for square matrix multiplication.

    We check:

        sum_t A_t[i,j] B_t[k,l] C_t[m,n]
        =
        delta_{n,i} delta_{j,k} delta_{l,m}

    over F_modulus.

    For 3x3 this checks 3^6 = 729 equations.
    For 2x2 this checks 2^6 = 64 equations.
    """
    n = matrix_size
    errors = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    for m in range(n):
                        for nu in range(n):
                            total = 0

                            for A, B, C in scheme:
                                total += (
                                    entry(A, n, i, j)
                                    * entry(B, n, k, ell)
                                    * entry(C, n, m, nu)
                                )

                            total %= modulus

                            target = (
                                1
                                if (nu == i and j == k and ell == m)
                                else 0
                            )

                            target %= modulus

                            if total != target:
                                errors.append(
                                    {
                                        "indices": (
                                            i + 1,
                                            j + 1,
                                            k + 1,
                                            ell + 1,
                                            m + 1,
                                            nu + 1,
                                        ),
                                        "got": total,
                                        "expected": target,
                                    }
                                )

                                if (
                                    max_errors is not None
                                    and len(errors) >= max_errors
                                ):
                                    return errors

    return errors


def verify_file(
    filename,
    matrix_size,
    modulus,
    transpose_c=False,
    max_errors_shown=5,
):
    """
    Verify one scheme file and return a summary dictionary.
    """
    scheme = read_scheme(
        filename,
        matrix_size=matrix_size,
        modulus=modulus,
        transpose_c=transpose_c,
    )

    errors = brent_errors(
        scheme,
        matrix_size=matrix_size,
        modulus=modulus,
        max_errors=None,
    )

    zero_rows = zero_factor_rows(scheme)

    first_error = ""

    if errors:
        first = errors[0]
        first_error = (
            f"indices={first['indices']} "
            f"got={first['got']} "
            f"expected={first['expected']}"
        )

    shown_errors = []

    for error in errors[:max_errors_shown]:
        shown_errors.append(
            f"indices={error['indices']} "
            f"got={error['got']} "
            f"expected={error['expected']}"
        )

    return {
        "file": str(filename),
        "rows": len(scheme),
        "valid_brent": len(errors) == 0,
        "number_of_errors": len(errors),
        "zero_factor_rows": ";".join(zero_rows),
        "first_error": first_error,
        "shown_errors": " | ".join(shown_errors),
    }


def collect_input_files(input_path, extension, recursive):
    """
    Return files to verify.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a file or folder: {input_path}")

    if not extension.startswith("."):
        extension = "." + extension

    if recursive:
        files = sorted(input_path.rglob(f"*{extension}"))
    else:
        files = sorted(input_path.glob(f"*{extension}"))

    return [path for path in files if path.is_file()]


def copy_to_folder(source_file, destination_folder):
    """
    Copy source_file into destination_folder.
    """
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    destination_file = destination_folder / Path(source_file).name
    shutil.copy2(source_file, destination_file)


def write_report(report_path, rows):
    """
    Write CSV report.
    """
    report_path = Path(report_path)

    fieldnames = [
        "file",
        "rows",
        "valid_brent",
        "number_of_errors",
        "zero_factor_rows",
        "first_error",
        "shown_errors",
    ]

    with open(report_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify compact-token matrix multiplication schemes using "
            "Brent equations over F_p."
        )
    )

    parser.add_argument(
        "input_path",
        help="Scheme file or folder of scheme files to verify.",
    )

    parser.add_argument(
        "--matrix-size",
        type=int,
        default=3,
        help="Matrix size. Use 3 for 3x3 schemes. Default: 3.",
    )

    parser.add_argument(
        "--modulus",
        type=int,
        default=2,
        help="Field modulus. Use 2 for F_2. Default: 2.",
    )

    parser.add_argument(
        "--extension",
        default="txt",
        help="File extension to check when input_path is a folder. Default: txt.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders too.",
    )

    parser.add_argument(
        "--transpose-c",
        action="store_true",
        help="Transpose C matrices while reading.",
    )

    parser.add_argument(
        "--report",
        default=None,
        help="Optional CSV report path.",
    )

    parser.add_argument(
        "--valid-dir",
        default=None,
        help="Optional folder where valid schemes are copied.",
    )

    parser.add_argument(
        "--invalid-dir",
        default=None,
        help="Optional folder where invalid schemes are copied.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files checked.",
    )

    args = parser.parse_args()

    files = collect_input_files(
        args.input_path,
        extension=args.extension,
        recursive=args.recursive,
    )

    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        print("No files found.")
        return

    print(f"Input path: {args.input_path}")
    print(f"Files to check: {len(files)}")
    print(f"Matrix size: {args.matrix_size}")
    print(f"Modulus: {args.modulus}")
    print()

    report_rows = []
    valid_count = 0
    invalid_count = 0
    error_count = 0

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Checking {file_path}")

        try:
            result = verify_file(
                file_path,
                matrix_size=args.matrix_size,
                modulus=args.modulus,
                transpose_c=args.transpose_c,
            )

            report_rows.append(result)

            if result["valid_brent"]:
                print("  VALID")
                valid_count += 1

                if args.valid_dir:
                    copy_to_folder(file_path, args.valid_dir)

            else:
                print(
                    f"  INVALID: {result['number_of_errors']} failed equations"
                )
                print(f"  First error: {result['first_error']}")
                invalid_count += 1

                if args.invalid_dir:
                    copy_to_folder(file_path, args.invalid_dir)

        except Exception as error:
            print(f"  ERROR: {error}")
            error_count += 1

            result = {
                "file": str(file_path),
                "rows": "",
                "valid_brent": False,
                "number_of_errors": "ERROR",
                "zero_factor_rows": "",
                "first_error": str(error),
                "shown_errors": "",
            }

            report_rows.append(result)

            if args.invalid_dir:
                copy_to_folder(file_path, args.invalid_dir)

        print()

    if args.report:
        write_report(args.report, report_rows)
        print(f"Report written to: {args.report}")

    print("Finished.")
    print(f"Valid schemes: {valid_count}")
    print(f"Invalid schemes: {invalid_count}")
    print(f"Files with read/check errors: {error_count}")


if __name__ == "__main__":
    main()
