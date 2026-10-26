from pathlib import Path
import argparse
import re
from collections import defaultdict


def clean_factor(text):
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def parse_linear_form(text, field):
    """
    Parses one linear form such as:
        a11+a12-a22
        b11+b12+b21+b22
        c11

    Returns a dictionary:
        {"a11": 1, "a12": 1, "a22": -1}
    """

    text = clean_factor(text)
    text = text.replace(" ", "")

    if text == "" or text == "0":
        return {}

    # Turn subtraction into +-
    text = text.replace("-", "+-")

    terms = text.split("+")
    coeffs = defaultdict(int)

    for term in terms:
        if term == "":
            continue

        sign = 1
        if term.startswith("-"):
            sign = -1
            term = term[1:]

        # Expected variable format: a11, b22, c12, etc.
        match = re.fullmatch(r"([abc])(\d)(\d)", term)
        if not match:
            raise ValueError(f"Could not parse term: {term}")

        var = term
        coeffs[var] += sign

    if field == "mod2":
        return {var: value % 2 for var, value in coeffs.items() if value % 2 != 0}

    return {var: value for var, value in coeffs.items() if value != 0}


def parse_scheme(file_path, field):
    """
    Parses the .exp scheme file.
    Each non-empty row should contain 3 factors separated by *.
    """

    rows = []

    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split("*")

            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_number} does not have exactly 3 factors separated by *:\n{line}"
                )

            a_form = parse_linear_form(parts[0], field)
            b_form = parse_linear_form(parts[1], field)
            c_form = parse_linear_form(parts[2], field)

            rows.append((a_form, b_form, c_form))

    return rows


def variable(kind, i, j):
    return f"{kind}{i}{j}"


def build_scheme_tensor(rows, field):
    """
    Expands the scheme:
        sum_t (A_t)(B_t)(C_t)

    Produces coefficients of monomials:
        aij * bkl * cmn
    """

    tensor = defaultdict(int)

    for a_form, b_form, c_form in rows:
        for a_var, a_coeff in a_form.items():
            for b_var, b_coeff in b_form.items():
                for c_var, c_coeff in c_form.items():
                    key = (a_var, b_var, c_var)
                    tensor[key] += a_coeff * b_coeff * c_coeff

    if field == "mod2":
        tensor = {
            key: value % 2
            for key, value in tensor.items()
            if value % 2 != 0
        }
    else:
        tensor = {
            key: value
            for key, value in tensor.items()
            if value != 0
        }

    return tensor


def build_target_tensor(n, m, p, c_convention):
    """
    Builds the true matrix multiplication tensor.

    Standard Brent convention:
        sum a_ij b_jk c_ki

    Output convention:
        sum a_ij b_jk c_ik
    """

    target = defaultdict(int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, p + 1):
                a_var = variable("a", i, j)
                b_var = variable("b", j, k)

                if c_convention == "brent":
                    c_var = variable("c", k, i)
                elif c_convention == "output":
                    c_var = variable("c", i, k)
                else:
                    raise ValueError("c_convention must be brent or output")

                target[(a_var, b_var, c_var)] = 1

    return target


def all_possible_keys(n, m, p):
    """
    For 2x2, this creates all 64 possible a*b*c monomials.
    """

    keys = []

    for ai in range(1, n + 1):
        for aj in range(1, m + 1):
            a_var = variable("a", ai, aj)

            for bi in range(1, m + 1):
                for bj in range(1, p + 1):
                    b_var = variable("b", bi, bj)

                    for ci in range(1, p + 1):
                        for cj in range(1, n + 1):
                            c_var = variable("c", ci, cj)

                            keys.append((a_var, b_var, c_var))

    return keys


def compare_tensors(scheme_tensor, target_tensor, n, m, p, field):
    mismatches = []

    for key in all_possible_keys(n, m, p):
        scheme_value = scheme_tensor.get(key, 0)
        target_value = target_tensor.get(key, 0)

        if field == "mod2":
            scheme_value %= 2
            target_value %= 2

        if scheme_value != target_value:
            mismatches.append((key, scheme_value, target_value))

    return mismatches


def check_file(file_path, n, m, p, field, c_convention):
    rows = parse_scheme(file_path, field)
    scheme_tensor = build_scheme_tensor(rows, field)
    target_tensor = build_target_tensor(n, m, p, c_convention)

    mismatches = compare_tensors(
        scheme_tensor=scheme_tensor,
        target_tensor=target_tensor,
        n=n,
        m=m,
        p=p,
        field=field,
    )

    return rows, mismatches


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Path to the .exp scheme file")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--field", choices=["mod2", "integer"], default="mod2")
    parser.add_argument("--c-convention", choices=["brent", "output", "auto"], default="auto")
    parser.add_argument("--show", type=int, default=20)

    args = parser.parse_args()

    file_path = Path(args.file).resolve()

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    conventions = ["brent", "output"] if args.c_convention == "auto" else [args.c_convention]

    print(f"Checking file: {file_path}")
    print(f"Dimensions: {args.n} x {args.m} x {args.p}")
    print(f"Field: {args.field}")
    print()

    for convention in conventions:
        rows, mismatches = check_file(
            file_path=file_path,
            n=args.n,
            m=args.m,
            p=args.p,
            field=args.field,
            c_convention=convention,
        )

        print(f"C convention tested: {convention}")
        print(f"Number of products / rows: {len(rows)}")

        if not mismatches:
            print("RESULT: CORRECT scheme")
        else:
            print("RESULT: INCORRECT scheme")
            print(f"Number of failed Brent equations: {len(mismatches)}")
            print()

            print(f"First {min(args.show, len(mismatches))} mismatches:")

            for key, scheme_value, target_value in mismatches[:args.show]:
                a_var, b_var, c_var = key
                print(
                    f"  {a_var} * {b_var} * {c_var}: "
                    f"scheme gives {scheme_value}, target is {target_value}"
                )

        print("-" * 60)


if __name__ == "__main__":
    main()