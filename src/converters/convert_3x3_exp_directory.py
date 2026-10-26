import re
import sys
import argparse
from pathlib import Path


def variable_order(letter, matrix_size):
    """
    Row-major order.

    For matrix_size = 3:
        a11, a12, a13,
        a21, a22, a23,
        a31, a32, a33
    """
    return [
        f"{letter}{row}{column}"
        for row in range(1, matrix_size + 1)
        for column in range(1, matrix_size + 1)
    ]


def clean_factor(factor):
    """
    Remove spaces and one pair of outer parentheses.
    """
    factor = factor.strip()

    if factor.startswith("(") and factor.endswith(")"):
        factor = factor[1:-1]

    return factor.replace(" ", "")


def factor_to_token(factor, expected_letter, matrix_size, modulus):
    """
    Convert one expression factor to a compact matrix token.

    Example over F_2, matrix_size = 3:
        a11-a12 -> 110000000

    Over F_2, +1 and -1 both become 1.
    Over F_3, -1 becomes 2.
    """
    factor = clean_factor(factor)
    order = variable_order(expected_letter, matrix_size)
    index_of = {variable: index for index, variable in enumerate(order)}

    token = [0] * (matrix_size * matrix_size)

    # Example matches:
    #   a11
    #   +a22
    #   -a31
    pattern = re.compile(
        rf"([+-]?)([abc][1-{matrix_size}][1-{matrix_size}])"
    )

    matches = list(pattern.finditer(factor))

    if not matches:
        raise ValueError(f"No variables found in factor: {factor}")

    # Check that the whole factor was parsed as signed variables.
    parsed_text = "".join(match.group(0) for match in matches)
    if parsed_text != factor:
        raise ValueError(
            f"Could not fully parse factor: {factor}. "
            f"Parsed only: {parsed_text}"
        )

    for match in matches:
        sign = match.group(1)
        variable = match.group(2)

        if not variable.startswith(expected_letter):
            raise ValueError(
                f"Expected {expected_letter}-variable but found "
                f"{variable} in factor {factor}"
            )

        coefficient = -1 if sign == "-" else 1
        index = index_of[variable]

        token[index] = (token[index] + coefficient) % modulus

    return "".join(str(value) for value in token)


def parse_scheme_line(line, matrix_size, modulus):
    """
    Convert one Exp-format line into normal_form_one.py token format.

    Input:
        (a11-a12)*(b11)*(c11-c21+c31)

    Output over F_2:
        110000000 100000000 100100100
    """
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    parts = line.split("*")

    if len(parts) != 3:
        raise ValueError(
            f"Line should have exactly 3 factors separated by *: {line}"
        )

    return (
        factor_to_token(parts[0], "a", matrix_size, modulus),
        factor_to_token(parts[1], "b", matrix_size, modulus),
        factor_to_token(parts[2], "c", matrix_size, modulus),
    )


def convert_one_file(input_path, output_path, matrix_size, modulus):
    converted_rows = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                parsed = parse_scheme_line(
                    line,
                    matrix_size=matrix_size,
                    modulus=modulus,
                )

                if parsed is not None:
                    converted_rows.append(parsed)

            except Exception as error:
                raise ValueError(
                    f"Error in file {input_path}, line {line_number}:\n"
                    f"{line.strip()}\n"
                    f"{error}"
                ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for A_token, B_token, C_token in converted_rows:
            output_file.write(f"{A_token} {B_token} {C_token}\n")

    return len(converted_rows)


def safe_flat_name(relative_path):
    """
    Convert nested path into a safe flat filename.

    Example:
        pattern/file.exp -> pattern__file.txt
    """
    parts = list(relative_path.with_suffix("").parts)
    return "__".join(parts) + ".txt"


def convert_directory(
    input_dir,
    output_dir,
    matrix_size=3,
    modulus=2,
    flat=False,
    max_files=None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    input_files = sorted(input_dir.rglob("*.exp"))

    if max_files is not None:
        input_files = input_files[:max_files]

    if not input_files:
        print(f"No .exp files found in {input_dir}")
        return

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Matrix size: {matrix_size}")
    print(f"Modulus: {modulus}")
    print(f"Flat output: {flat}")
    print(f"Files found: {len(input_files)}")
    print()

    success_count = 0
    error_count = 0

    for index, input_path in enumerate(input_files, start=1):
        relative_path = input_path.relative_to(input_dir)

        if flat:
            output_path = output_dir / safe_flat_name(relative_path)
        else:
            output_path = (output_dir / relative_path).with_suffix(".txt")

        try:
            row_count = convert_one_file(
                input_path,
                output_path,
                matrix_size=matrix_size,
                modulus=modulus,
            )

            print(
                f"[{index}/{len(input_files)}] Converted "
                f"{relative_path} -> {output_path.relative_to(output_dir)} "
                f"({row_count} rows)"
            )
            success_count += 1

        except Exception as error:
            print(f"[{index}/{len(input_files)}] ERROR {relative_path}: {error}")
            error_count += 1

    print()
    print("Finished.")
    print(f"Successful files: {success_count}")
    print(f"Files with errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Heule-Kauers-Seidl 3x3 Exp-format schemes "
            "to compact matrix-token format for normal_form_one.py."
        )
    )

    parser.add_argument(
        "input_dir",
        help="Folder containing .exp files, possibly inside subfolders.",
    )

    parser.add_argument(
        "output_dir",
        help="Folder where converted .txt files should be written.",
    )

    parser.add_argument(
        "--matrix-size",
        type=int,
        default=3,
        help="Matrix size. Default: 3.",
    )

    parser.add_argument(
        "--modulus",
        type=int,
        default=2,
        help="Field modulus. Use 2 for Z2/F2. Default: 2.",
    )

    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write all converted files into one folder instead of preserving subfolders.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    args = parser.parse_args()

    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        matrix_size=args.matrix_size,
        modulus=args.modulus,
        flat=args.flat,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
