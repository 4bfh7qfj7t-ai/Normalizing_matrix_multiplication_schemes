import re
import sys
from pathlib import Path


# ============================================================
# Variable order
# ============================================================
# We use row-major order:
#
#   11, 12, 21, 22
#
# So:
#
#   a11 + a22  -> 1001
#   a12 + a21  -> 0110
#   c11 + c12 + c21 + c22 -> 1111
# ============================================================

A_ORDER = ["a11", "a12", "a21", "a22"]
B_ORDER = ["b11", "b12", "b21", "b22"]
C_ORDER = ["c11", "c12", "c21", "c22"]


def clean_factor(factor):
    """
    Remove spaces and outer parentheses.

    Example:
        "(a11 + a22)" -> "a11+a22"
    """
    factor = factor.strip()

    if factor.startswith("(") and factor.endswith(")"):
        factor = factor[1:-1]

    factor = factor.replace(" ", "")

    return factor


def factor_to_bits(factor, expected_letter):
    """
    Convert one factor into a 4-bit vector.

    Example:
        a11+a22 -> 1001

    The expected_letter tells us whether we expect:
        a variables, b variables, or c variables.
    """

    factor = clean_factor(factor)

    if expected_letter == "a":
        order = A_ORDER
    elif expected_letter == "b":
        order = B_ORDER
    elif expected_letter == "c":
        order = C_ORDER
    else:
        raise ValueError("expected_letter must be 'a', 'b', or 'c'")

    bits = [0, 0, 0, 0]

    # Find variables like a11, b22, c12, etc.
    variables = re.findall(r"[abc][12][12]", factor)

    for var in variables:
        if not var.startswith(expected_letter):
            raise ValueError(
                f"Expected {expected_letter}-variable but found {var} in factor {factor}"
            )

        index = order.index(var)

        # Since we are working over Z2:
        # x + x = 0
        #
        # So if a variable appears twice, it cancels.
        bits[index] ^= 1

    return "".join(str(bit) for bit in bits)


def parse_scheme_line(line):
    """
    Convert one human-readable scheme line into 4-bit vector format.

    Input:
        (a11 + a22)*(b11 + b22)*(c22)

    Output:
        ("1001", "1001", "0001")
    """

    line = line.strip()

    if not line:
        return None

    if line.startswith("#"):
        return None

    parts = line.split("*")

    if len(parts) != 3:
        raise ValueError(f"Line should have exactly 3 factors separated by *: {line}")

    A_factor, B_factor, C_factor = parts

    A_bits = factor_to_bits(A_factor, "a")
    B_bits = factor_to_bits(B_factor, "b")
    C_bits = factor_to_bits(C_factor, "c")

    return A_bits, B_bits, C_bits


def convert_one_file(input_path, output_path):
    """
    Convert one raw scheme file into 4-bit vector format.
    """

    converted_rows = []

    with open(input_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                parsed = parse_scheme_line(line)

                if parsed is not None:
                    converted_rows.append(parsed)

            except Exception as e:
                raise ValueError(
                    f"Error in file {input_path}, line {line_number}:\n"
                    f"{line.strip()}\n"
                    f"{e}"
                )

    with open(output_path, "w") as f:
        for A_bits, B_bits, C_bits in converted_rows:
            f.write(f"{A_bits} {B_bits} {C_bits}\n")

    return len(converted_rows)


def convert_directory(input_dir, output_dir):
    """
    Read every .txt file in input_dir and create a converted file
    in output_dir.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.exp"))

    if not input_files:
        print(f"No .exp files found in {input_dir}")
        return

    print(f"Found {len(input_files)} text files.\n")

    for input_path in input_files:
        output_filename = input_path.stem + "_4bit.txt"
        output_path = output_dir / output_filename

        row_count = convert_one_file(input_path, output_path)

        print(f"Converted: {input_path.name}")
        print(f"Rows written: {row_count}")
        print(f"Output file: {output_path}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python3 convert_directory.py <input_directory> <output_directory>")
        print()
        print("Example:")
        print("python3 convert_directory.py schemes_raw schemes_4bit")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    convert_directory(input_directory, output_directory)