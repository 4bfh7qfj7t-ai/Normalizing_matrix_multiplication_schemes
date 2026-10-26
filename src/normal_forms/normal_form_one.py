from itertools import permutations
from pathlib import Path
import argparse

from gl_cube import GLCube


# ============================================================
# Ordering for normal form
# ============================================================


def matrix_key(matrix, matrix_space):
    """
    Return the colexicographic ordering key for one matrix.
    """
    return matrix_space.colex_key(matrix)


def row_key(row, matrix_space):
    """
    Compare a scheme row (A, B, C), using A first, then B, then C.
    """
    A, B, C = row

    return (
        matrix_key(A, matrix_space)
        + matrix_key(B, matrix_space)
        + matrix_key(C, matrix_space)
    )


def scheme_key(scheme, matrix_space):
    """
    Compare schemes row by row.
    """
    key = []

    for row in scheme:
        key.extend(row_key(row, matrix_space))

    return tuple(key)


def canonical_tail(rows, matrix_space):
    """
    Sort rows because multiplication terms may be permuted.
    """
    return tuple(
        sorted(
            rows,
            key=lambda row: row_key(
                row,
                matrix_space,
            ),
        )
    )


# ============================================================
# Rank pattern
# ============================================================


def rank_vector(row, matrix_space):
    """
    Return the ranks of A, B, and C over the current field.
    """
    A, B, C = row

    return (
        matrix_space.rank(A),
        matrix_space.rank(B),
        matrix_space.rank(C),
    )


def rank_pattern(scheme, matrix_space):
    """
    Return the sorted rank vectors of the scheme.
    """
    return tuple(
        sorted(
            (
                rank_vector(row, matrix_space)
                for row in scheme
            ),
            reverse=True,
        )
    )


# ============================================================
# Column-permutation symmetry
# ============================================================


def permutation_parity(permutation):
    """
    Return 0 for an even permutation and 1 for an odd permutation.
    """
    inversions = 0

    for first in range(len(permutation)):
        for second in range(first + 1, len(permutation)):
            if permutation[first] > permutation[second]:
                inversions += 1

    return inversions % 2


def apply_column_permutation_to_row(
    row,
    permutation,
    matrix_space,
):
    """
    Permute A, B, and C.

    For an odd permutation, transpose all three matrices.
    """
    new_row = tuple(
        row[index]
        for index in permutation
    )

    if permutation_parity(permutation) == 1:
        new_row = tuple(
            matrix_space.transpose(matrix)
            for matrix in new_row
        )

    return new_row


def apply_column_permutation_to_scheme(
    scheme,
    permutation,
    matrix_space,
):
    """
    Apply one column permutation to every row of a scheme.
    """
    return tuple(
        apply_column_permutation_to_row(
            row,
            permutation,
            matrix_space,
        )
        for row in scheme
    )


# ============================================================
# Reading, writing, and displaying schemes
# ============================================================


def read_scheme(
    filename,
    matrix_space,
    transpose_c=False,
):
    """
    Read one multiplication scheme.

    Every nonblank, noncomment line must contain three matrix tokens:

        A B C

    Examples for 2x2 matrices:

        Over F_2: 1001 1001 0001
        Over F_3: 1021 2101 0012
    """
    scheme = []

    with open(filename, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_number} must contain exactly "
                    f"three matrices: {line}"
                )

            A = matrix_space.parse(parts[0])
            B = matrix_space.parse(parts[1])
            C = matrix_space.parse(parts[2])

            if transpose_c:
                C = matrix_space.transpose(C)

            scheme.append((A, B, C))

    if not scheme:
        raise ValueError(
            f"The input file contains no scheme rows: {filename}"
        )

    return tuple(scheme)


def write_scheme(
    filename,
    scheme,
    matrix_space,
):
    """
    Write one scheme using the current MatrixSpace format.
    """
    with open(filename, "w", encoding="utf-8") as output_file:
        for A, B, C in scheme:
            output_file.write(
                f"{matrix_space.format(A)} "
                f"{matrix_space.format(B)} "
                f"{matrix_space.format(C)}\n"
            )


def print_scheme(scheme, matrix_space):
    """
    Print one scheme using the current field and matrix size.
    """
    for A, B, C in scheme:
        print(
            matrix_space.format(A),
            matrix_space.format(B),
            matrix_space.format(C),
        )


def remove_one_row(rows, row_to_remove):
    """
    Remove one occurrence of row_to_remove.

    Only one copy is removed because duplicate rows are possible.
    """
    output = []
    removed = False

    for row in rows:
        if not removed and row == row_to_remove:
            removed = True
            continue

        output.append(row)

    if not removed:
        raise ValueError(
            "Could not remove the selected row from the transformed tail."
        )

    return tuple(output)


# ============================================================
# Algorithm 1
# ============================================================


def normal_form_algorithm_1(
    scheme,
    gl_cube,
    verbose=False,
):
    """
    Compute the Algorithm 1 normal form using the supplied GLCube.

    The algorithm itself does not choose a field or matrix size.
    Those choices are contained in gl_cube.
    """
    matrix_space = gl_cube.matrix_space

    # Step 1:
    # Consider all six permutations of the three tensor factors.
    # Keep only those with maximal rank pattern.
    permuted_candidates = []

    for permutation in permutations(range(3)):
        permuted_scheme = apply_column_permutation_to_scheme(
            scheme,
            permutation,
            matrix_space,
        )

        permuted_scheme = canonical_tail(
            permuted_scheme,
            matrix_space,
        )

        permuted_candidates.append(
            (
                rank_pattern(
                    permuted_scheme,
                    matrix_space,
                ),
                permuted_scheme,
            )
        )

    maximal_rank_pattern = max(
        pattern
        for pattern, _ in permuted_candidates
    )

    candidates = [
        permuted_scheme
        for pattern, permuted_scheme in permuted_candidates
        if pattern == maximal_rank_pattern
    ]

    if verbose:
        print(
            f"GL({gl_cube.matrix_size}, "
            f"F_{gl_cube.modulus}) size:",
            gl_cube.base_size,
        )

        print(
            f"GL({gl_cube.matrix_size}, "
            f"F_{gl_cube.modulus})^3 size:",
            len(gl_cube),
        )

        print(
            "Number of column-permutation candidates:",
            len(candidates),
        )

        print(
            "Maximal rank pattern:",
            maximal_rank_pattern,
        )

        print()

    best_normal_form = None

    # Step 2:
    # Build a normal-form candidate from each surviving
    # column-permutation candidate.
    for candidate_index, seed_scheme in enumerate(
        candidates,
        start=1,
    ):
        candidate = []
        tails = {seed_scheme}

        # Initially every element of GL(n, F_p)^3 is allowed.
        # GLCube generates these elements lazily.
        stabilizer = gl_cube

        if verbose:
            print(
                f"Starting candidate "
                f"{candidate_index}/{len(candidates)}"
            )

        while tails != {()}:
            current_min_row = None
            new_tails = set()

            for tail in tails:
                maximal_row_rank = max(
                    rank_vector(row, matrix_space)
                    for row in tail
                )

                rows_to_try = [
                    row
                    for row in tail
                    if rank_vector(
                        row,
                        matrix_space,
                    ) == maximal_row_rank
                ]

                for row in rows_to_try:
                    # Find the smallest image of this row under
                    # the current stabilizer in one pass.
                    #
                    # We retain only transformations achieving
                    # the minimum rather than storing every image.
                    local_min_key = None
                    best_pairs = []

                    for group_element in stabilizer:
                        transformed_row = gl_cube.apply_to_row(
                            row,
                            group_element,
                        )

                        transformed_key = row_key(
                            transformed_row,
                            matrix_space,
                        )

                        if (
                            local_min_key is None
                            or transformed_key < local_min_key
                        ):
                            local_min_key = transformed_key
                            best_pairs = [
                                (
                                    transformed_row,
                                    group_element,
                                )
                            ]

                        elif transformed_key == local_min_key:
                            best_pairs.append(
                                (
                                    transformed_row,
                                    group_element,
                                )
                            )

                    if not best_pairs:
                        raise RuntimeError(
                            "The current stabilizer contained "
                            "no transformations."
                        )

                    local_min_row = best_pairs[0][0]

                    local_row_key = row_key(
                        local_min_row,
                        matrix_space,
                    )

                    if (
                        current_min_row is None
                        or local_row_key
                        < row_key(
                            current_min_row,
                            matrix_space,
                        )
                    ):
                        current_min_row = local_min_row
                        new_tails = set()

                    if local_row_key == row_key(
                        current_min_row,
                        matrix_space,
                    ):
                        for (
                            transformed_row,
                            group_element,
                        ) in best_pairs:
                            transformed_tail = (
                                gl_cube.apply_to_scheme(
                                    tail,
                                    group_element,
                                )
                            )

                            remaining_tail = remove_one_row(
                                transformed_tail,
                                transformed_row,
                            )

                            new_tails.add(
                                canonical_tail(
                                    remaining_tail,
                                    matrix_space,
                                )
                            )

            if current_min_row is None:
                raise RuntimeError(
                    "Algorithm 1 could not select the next row."
                )

            # Restrict future transformations to those that fix
            # every row already selected.
            stabilizer = gl_cube.stabilizer(
                current_min_row,
                within=stabilizer,
            )

            candidate.append(current_min_row)
            tails = new_tails

            if verbose:
                row_as_text = tuple(
                    matrix_space.format(matrix)
                    for matrix in current_min_row
                )

                print("Added row:", row_as_text)
                print("Remaining tails:", len(tails))
                print("Stabilizer size:", len(stabilizer))
                print()

        candidate = tuple(candidate)

        if (
            best_normal_form is None
            or scheme_key(
                candidate,
                matrix_space,
            )
            < scheme_key(
                best_normal_form,
                matrix_space,
            )
        ):
            best_normal_form = candidate

    return best_normal_form


# ============================================================
# Main program
# ============================================================


def default_output_name(
    input_file,
    output_dir=None,
):
    """
    Construct the default output filename.
    """
    input_path = Path(input_file)

    output_filename = (
        input_path.stem
        + "_normal_form.txt"
    )

    if output_dir is None:
        return str(
            input_path.with_name(output_filename)
        )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return str(
        output_path / output_filename
    )


def build_argument_parser():
    """
    Create the command-line parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Algorithm 1 normal form for one "
            "matrix multiplication scheme over a prime field."
        )
    )

    parser.add_argument(
        "input_file",
        help=(
            "Input scheme file. Every row must contain "
            "three matrix tokens: A B C."
        ),
    )

    parser.add_argument(
        "--modulus",
        type=int,
        default=2,
        help=(
            "Prime field modulus. Use 2 for F_2 or "
            "3 for F_3. Default: 2."
        ),
    )

    parser.add_argument(
        "--matrix-size",
        type=int,
        default=2,
        help=(
            "Number of rows and columns in each square "
            "factor matrix. Default: 2."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output file. If omitted, a default "
            "*_normal_form.txt file is created."
        ),
    )

    parser.add_argument(
        "--output-dir",
        help=(
            "Folder in which the normal-form file "
            "should be written."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print Algorithm 1 progress row by row."
        ),
    )

    parser.add_argument(
        "--transpose-c",
        action="store_true",
        help=(
            "Transpose C while reading. This is useful "
            "for the C^T = AB convention."
        ),
    )

    return parser


def main():
    """
    Run the command-line program.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.output and args.output_dir:
        parser.error(
            "Use either --output or --output-dir, not both."
        )

    try:
        gl_cube = GLCube(
            modulus=args.modulus,
            matrix_size=args.matrix_size,
        )

    except ValueError as error:
        parser.error(str(error))

    matrix_space = gl_cube.matrix_space

    output_file = (
        args.output
        or default_output_name(
            args.input_file,
            output_dir=args.output_dir,
        )
    )

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Using group: {gl_cube.name}")
    print(f"Base group size: {gl_cube.base_size}")
    print(f"Cube size: {len(gl_cube)}")
    print()

    if args.modulus != 2:
        print(
            "Warning: field arithmetic and the "
            "GL(n, F_p)^3 action are generalized, "
            "but independent row-scaling normalization "
            "has not yet been added."
        )
        print()

    try:
        scheme = read_scheme(
            args.input_file,
            matrix_space,
            transpose_c=args.transpose_c,
        )

    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("Input scheme:")
    print_scheme(
        scheme,
        matrix_space,
    )
    print()

    print("Input rank pattern:")
    print(
        rank_pattern(
            scheme,
            matrix_space,
        )
    )
    print()

    normal_form = normal_form_algorithm_1(
        scheme,
        gl_cube=gl_cube,
        verbose=args.verbose,
    )

    print("Normal form:")
    print_scheme(
        normal_form,
        matrix_space,
    )
    print()

    print("Normal form rank pattern:")
    print(
        rank_pattern(
            normal_form,
            matrix_space,
        )
    )
    print()

    write_scheme(
        output_file,
        normal_form,
        matrix_space,
    )

    print(
        f"Normal form written to: {output_file}"
    )


if __name__ == "__main__":
    main()