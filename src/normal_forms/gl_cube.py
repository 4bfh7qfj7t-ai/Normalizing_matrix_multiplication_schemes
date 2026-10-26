"""Finite-field matrix arithmetic and GL(n, F_p)^3.

This module contains the field arithmetic, square-matrix operations,
the general linear group GL(n, F_p), and its three-factor sandwich action.

The normal-form algorithm should use a GLCube object instead of
hard-coding GL(2, F_2)^3.
"""

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence


# A square matrix is stored in row-major order as a flat tuple.
#
# For a 2x2 matrix:
#
#     [a b]
#     [c d]
#
# the stored form is:
#
#     (a, b, c, d)
Matrix = tuple[int, ...]

# One multiplication-scheme row contains three matrices.
SchemeRow = tuple[Matrix, Matrix, Matrix]

# A scheme is a tuple of rows.
Scheme = tuple[SchemeRow, ...]


def is_prime(value: int) -> bool:
    """Return True exactly when value is prime."""

    if value < 2:
        return False

    if value == 2:
        return True

    if value % 2 == 0:
        return False

    divisor = 3

    while divisor * divisor <= value:
        if value % divisor == 0:
            return False

        divisor += 2

    return True


@dataclass(frozen=True)
class PrimeField:
    """Arithmetic in the prime field F_p."""

    modulus: int

    def __post_init__(self) -> None:
        if not is_prime(self.modulus):
            raise ValueError(
                f"Field modulus must be prime, got {self.modulus}."
            )

    def normalize(self, value: int) -> int:
        """Return the representative from 0 to p-1."""

        return value % self.modulus

    def add(self, left: int, right: int) -> int:
        return (left + right) % self.modulus

    def subtract(self, left: int, right: int) -> int:
        return (left - right) % self.modulus

    def multiply(self, left: int, right: int) -> int:
        return (left * right) % self.modulus

    def negate(self, value: int) -> int:
        return (-value) % self.modulus

    def inverse(self, value: int) -> int:
        """Return the multiplicative inverse of a nonzero field element."""

        value %= self.modulus

        if value == 0:
            raise ZeroDivisionError(
                "Zero has no multiplicative inverse."
            )

        return pow(value, -1, self.modulus)


@dataclass(frozen=True)
class MatrixSpace:
    """Square n by n matrices over a prime field."""

    size: int
    field: PrimeField

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(
                "Matrix size must be at least 1."
            )

    @property
    def entry_count(self) -> int:
        return self.size * self.size

    @property
    def zero(self) -> Matrix:
        return (0,) * self.entry_count

    @property
    def identity(self) -> Matrix:
        return tuple(
            1 if row == column else 0
            for row in range(self.size)
            for column in range(self.size)
        )

    def coerce(self, entries: Sequence[int]) -> Matrix:
        """
        Validate a matrix and reduce all its entries modulo p.
        """

        if len(entries) != self.entry_count:
            raise ValueError(
                f"Expected {self.entry_count} entries, "
                f"got {len(entries)}."
            )

        return tuple(
            self.field.normalize(value)
            for value in entries
        )

    def parse(self, token: str) -> Matrix:
        """
        Parse a compact matrix token.

        For 2x2 matrices over F_2 or F_3:

            1001 -> (1, 0, 0, 1)
            1021 -> (1, 0, 2, 1)

        Comma-separated entries are also accepted:

            1,0,2,1
        """

        token = token.strip()

        if not token:
            raise ValueError("Matrix token is empty.")

        if "," in token:
            parts = token.split(",")

        else:
            if self.field.modulus > 10:
                raise ValueError(
                    "Use comma-separated entries when "
                    "the modulus is greater than 10."
                )

            parts = list(token)

        try:
            entries = tuple(int(part) for part in parts)

        except ValueError as error:
            raise ValueError(
                f"Invalid matrix token: {token}"
            ) from error

        if len(entries) != self.entry_count:
            raise ValueError(
                f"Matrix token must contain "
                f"{self.entry_count} entries: {token}"
            )

        for value in entries:
            if value < 0 or value >= self.field.modulus:
                raise ValueError(
                    f"Entry {value} is outside F_"
                    f"{self.field.modulus}: {token}"
                )

        return entries

    def format(self, matrix: Matrix) -> str:
        """Convert a matrix back into compact text."""

        matrix = self.coerce(matrix)

        separator = (
            ""
            if self.field.modulus <= 10
            else ","
        )

        return separator.join(
            str(value)
            for value in matrix
        )

    def transpose(self, matrix: Matrix) -> Matrix:
        matrix = self.coerce(matrix)
        n = self.size

        return tuple(
            matrix[column * n + row]
            for row in range(n)
            for column in range(n)
        )

    def multiply(
        self,
        left: Matrix,
        right: Matrix,
    ) -> Matrix:
        """Multiply two matrices over the current field."""

        left = self.coerce(left)
        right = self.coerce(right)

        n = self.size
        p = self.field.modulus

        output = [0] * self.entry_count

        for row in range(n):
            for column in range(n):
                total = 0

                for middle in range(n):
                    total += (
                        left[row * n + middle]
                        * right[middle * n + column]
                    )

                output[row * n + column] = total % p

        return tuple(output)

    def _row_rank(
        self,
        rows: Sequence[Sequence[int]],
    ) -> int:
        """
        Compute rank using Gaussian elimination over F_p.
        """

        if not rows:
            return 0

        p = self.field.modulus

        work = [
            [value % p for value in row]
            for row in rows
        ]

        row_count = len(work)
        column_count = len(work[0])
        pivot_row = 0

        for column in range(column_count):
            pivot = None

            for row in range(pivot_row, row_count):
                if work[row][column] != 0:
                    pivot = row
                    break

            if pivot is None:
                continue

            work[pivot_row], work[pivot] = (
                work[pivot],
                work[pivot_row],
            )

            pivot_inverse = self.field.inverse(
                work[pivot_row][column]
            )

            work[pivot_row] = [
                value * pivot_inverse % p
                for value in work[pivot_row]
            ]

            for row in range(row_count):
                if row == pivot_row:
                    continue

                multiplier = work[row][column]

                if multiplier == 0:
                    continue

                work[row] = [
                    (
                        work[row][index]
                        - multiplier
                        * work[pivot_row][index]
                    ) % p
                    for index in range(column_count)
                ]

            pivot_row += 1

            if pivot_row == row_count:
                break

        return pivot_row

    def rank(self, matrix: Matrix) -> int:
        """Return the rank of one square matrix."""

        matrix = self.coerce(matrix)
        n = self.size

        rows = [
            list(matrix[row * n:(row + 1) * n])
            for row in range(n)
        ]

        return self._row_rank(rows)

    def is_invertible(self, matrix: Matrix) -> bool:
        return self.rank(matrix) == self.size

    def inverse(self, matrix: Matrix) -> Matrix:
        """
        Compute a matrix inverse using augmented Gaussian elimination.
        """

        matrix = self.coerce(matrix)

        n = self.size
        p = self.field.modulus
        identity = self.identity

        rows = []

        for row in range(n):
            left_side = list(
                matrix[row * n:(row + 1) * n]
            )

            right_side = list(
                identity[row * n:(row + 1) * n]
            )

            rows.append(left_side + right_side)

        for column in range(n):
            pivot = None

            for row in range(column, n):
                if rows[row][column] != 0:
                    pivot = row
                    break

            if pivot is None:
                raise ValueError(
                    f"Matrix is not invertible: {matrix}"
                )

            rows[column], rows[pivot] = (
                rows[pivot],
                rows[column],
            )

            pivot_inverse = self.field.inverse(
                rows[column][column]
            )

            rows[column] = [
                value * pivot_inverse % p
                for value in rows[column]
            ]

            for row in range(n):
                if row == column:
                    continue

                multiplier = rows[row][column]

                if multiplier == 0:
                    continue

                rows[row] = [
                    (
                        rows[row][index]
                        - multiplier
                        * rows[column][index]
                    ) % p
                    for index in range(2 * n)
                ]

        return tuple(
            rows[row][n + column]
            for row in range(n)
            for column in range(n)
        )

    def colex_key(
        self,
        matrix: Matrix,
    ) -> tuple[int, ...]:
        """
        Colexicographic ordering by columns.

        For a 2x2 matrix:

            [a b]
            [c d]

        this returns:

            (b, d, a, c)

        which matches the original program.
        """

        matrix = self.coerce(matrix)
        n = self.size

        return tuple(
            matrix[row * n + column]
            for column in reversed(range(n))
            for row in range(n)
        )

    def invertible_matrices(
        self,
    ) -> tuple[Matrix, ...]:
        """
        Generate every matrix in GL(n, F_p).

        Rows are added only when they increase the current rank.
        """

        p = self.field.modulus
        n = self.size

        possible_rows = tuple(
            product(range(p), repeat=n)
        )

        output = []

        def extend(
            rows: tuple[tuple[int, ...], ...]
        ) -> None:
            if len(rows) == n:
                output.append(
                    tuple(
                        value
                        for row in rows
                        for value in row
                    )
                )
                return

            required_rank = len(rows) + 1

            for possible_row in possible_rows:
                candidate = rows + (
                    tuple(possible_row),
                )

                if (
                    self._row_rank(candidate)
                    == required_rank
                ):
                    extend(candidate)

        extend(())

        return tuple(output)


@dataclass(frozen=True)
class GLCubeElement:
    """
    One element (U, V, W) of GL(n, F_p)^3.

    The inverses are stored with the matrices so that they do not
    need to be recalculated every time the action is applied.
    """

    U: Matrix
    V: Matrix
    W: Matrix

    U_inverse: Matrix
    V_inverse: Matrix
    W_inverse: Matrix


class GLCube:
    """
    The group GL(n, F_p)^3 acting by sandwich transformations.

    On one scheme row (A, B, C), the action is:

        A -> U A V^-1
        B -> V B W^-1
        C -> W C U^-1
    """

    def __init__(
        self,
        modulus: int,
        matrix_size: int,
    ) -> None:
        self.field = PrimeField(modulus)

        self.matrix_space = MatrixSpace(
            matrix_size,
            self.field,
        )

        self._base_group = (
            self.matrix_space.invertible_matrices()
        )

        self._inverse_map = {
            matrix: self.matrix_space.inverse(matrix)
            for matrix in self._base_group
        }

    @property
    def modulus(self) -> int:
        return self.field.modulus

    @property
    def matrix_size(self) -> int:
        return self.matrix_space.size

    @property
    def base_group(self) -> tuple[Matrix, ...]:
        """
        Return GL(n, F_p), not its cube.
        """

        return self._base_group

    @property
    def base_size(self) -> int:
        return len(self._base_group)

    @property
    def cube_size(self) -> int:
        return self.base_size ** 3

    @property
    def name(self) -> str:
        return (
            f"GL({self.matrix_size}, "
            f"F_{self.modulus})^3"
        )

    def __len__(self) -> int:
        return self.cube_size

    def __iter__(self) -> Iterator[GLCubeElement]:
        """
        Generate cube elements lazily.

        The complete cube is not permanently stored in memory.
        """

        for U in self._base_group:
            U_inverse = self._inverse_map[U]

            for V in self._base_group:
                V_inverse = self._inverse_map[V]

                for W in self._base_group:
                    yield GLCubeElement(
                        U=U,
                        V=V,
                        W=W,
                        U_inverse=U_inverse,
                        V_inverse=V_inverse,
                        W_inverse=self._inverse_map[W],
                    )

    def apply_to_row(
        self,
        row: SchemeRow,
        element: GLCubeElement,
    ) -> SchemeRow:
        """Apply one cube element to one scheme row."""

        A, B, C = row
        multiply = self.matrix_space.multiply

        A_new = multiply(
            multiply(element.U, A),
            element.V_inverse,
        )

        B_new = multiply(
            multiply(element.V, B),
            element.W_inverse,
        )

        C_new = multiply(
            multiply(element.W, C),
            element.U_inverse,
        )

        return (A_new, B_new, C_new)

    def apply_to_scheme(
        self,
        scheme: Scheme,
        element: GLCubeElement,
    ) -> Scheme:
        """Apply one cube element to every row."""

        return tuple(
            self.apply_to_row(row, element)
            for row in scheme
        )

    def stabilizer(
        self,
        row: SchemeRow,
        within: Iterable[GLCubeElement] = None,
    ) -> tuple[GLCubeElement, ...]:
        """
        Return elements that leave the specified row unchanged.

        If 'within' is supplied, search only inside that collection.
        """

        source = self if within is None else within

        return tuple(
            element
            for element in source
            if self.apply_to_row(row, element) == row
        )