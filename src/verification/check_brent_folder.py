from pathlib import Path
import argparse
import csv
import shutil

from check_brent import check_file


def check_one_file(file_path, n, m, p, field, conventions):
    """
    Check one .exp file under one or more C conventions.
    Returns a dictionary with status and details.
    """

    best_failed_count = None
    best_convention = ""
    row_count = ""

    for convention in conventions:
        rows, mismatches = check_file(
            file_path=file_path,
            n=n,
            m=m,
            p=p,
            field=field,
            c_convention=convention,
        )

        row_count = len(rows)
        failed_count = len(mismatches)

        if best_failed_count is None or failed_count < best_failed_count:
            best_failed_count = failed_count
            best_convention = convention

        if failed_count == 0:
            return {
                "status": "CORRECT",
                "row_count": row_count,
                "correct_convention": convention,
                "failed_brent_equations": 0,
                "best_convention": convention,
                "best_failed_count": 0,
                "error": "",
            }

    return {
        "status": "INCORRECT",
        "row_count": row_count,
        "correct_convention": "",
        "failed_brent_equations": best_failed_count,
        "best_convention": best_convention,
        "best_failed_count": best_failed_count,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        help="Folder containing .exp scheme files to check"
    )

    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)

    parser.add_argument(
        "--field",
        choices=["mod2", "integer"],
        default="mod2"
    )

    parser.add_argument(
        "--c-convention",
        choices=["brent", "output", "auto"],
        default="auto"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Check .exp files inside subfolders too"
    )

    parser.add_argument(
        "--csv",
        default="brent_folder_summary.csv",
        help="Output CSV summary file"
    )

    parser.add_argument(
        "--copy-correct",
        default=None,
        help="Optional folder where correct schemes will be copied"
    )

    parser.add_argument(
        "--copy-incorrect",
        default=None,
        help="Optional folder where incorrect schemes will be copied"
    )

    args = parser.parse_args()

    folder = Path(args.folder).resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if args.recursive:
        files = sorted(folder.rglob("*.exp"))
    else:
        files = sorted(folder.glob("*.exp"))

    if args.c_convention == "auto":
        conventions = ["brent", "output"]
    else:
        conventions = [args.c_convention]

    copy_correct_dir = None
    if args.copy_correct:
        copy_correct_dir = Path(args.copy_correct).resolve()
        copy_correct_dir.mkdir(parents=True, exist_ok=True)

    copy_incorrect_dir = None
    if args.copy_incorrect:
        copy_incorrect_dir = Path(args.copy_incorrect).resolve()
        copy_incorrect_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    correct_count = 0
    incorrect_count = 0
    error_count = 0

    print(f"Checking folder: {folder}")
    print(f"Total .exp files found: {len(files)}")
    print(f"Dimensions: {args.n} x {args.m} x {args.p}")
    print(f"Field: {args.field}")
    print(f"C convention: {args.c_convention}")
    print()

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path.name}")

        try:
            result = check_one_file(
                file_path=file_path,
                n=args.n,
                m=args.m,
                p=args.p,
                field=args.field,
                conventions=conventions,
            )

            status = result["status"]

            if status == "CORRECT":
                correct_count += 1
                print(
                    f"  CORRECT | rows={result['row_count']} | "
                    f"convention={result['correct_convention']}"
                )

                if copy_correct_dir is not None:
                    shutil.copy(file_path, copy_correct_dir / file_path.name)

            else:
                incorrect_count += 1
                print(
                    f"  INCORRECT | rows={result['row_count']} | "
                    f"best convention={result['best_convention']} | "
                    f"failed equations={result['best_failed_count']}"
                )

                if copy_incorrect_dir is not None:
                    shutil.copy(file_path, copy_incorrect_dir / file_path.name)

            summary_rows.append({
                "file_name": file_path.name,
                "path": str(file_path),
                "status": status,
                "row_count": result["row_count"],
                "correct_convention": result["correct_convention"],
                "failed_brent_equations": result["failed_brent_equations"],
                "best_convention": result["best_convention"],
                "best_failed_count": result["best_failed_count"],
                "error": "",
            })

        except Exception as e:
            error_count += 1
            print(f"  ERROR: {e}")

            summary_rows.append({
                "file_name": file_path.name,
                "path": str(file_path),
                "status": "ERROR",
                "row_count": "",
                "correct_convention": "",
                "failed_brent_equations": "",
                "best_convention": "",
                "best_failed_count": "",
                "error": str(e),
            })

    csv_path = Path(args.csv).resolve()

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "file_name",
            "path",
            "status",
            "row_count",
            "correct_convention",
            "failed_brent_equations",
            "best_convention",
            "best_failed_count",
            "error",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("Finished.")
    print(f"Correct schemes: {correct_count}")
    print(f"Incorrect schemes: {incorrect_count}")
    print(f"Errors: {error_count}")
    print(f"Summary CSV: {csv_path}")

    if copy_correct_dir is not None:
        print(f"Correct schemes copied to: {copy_correct_dir}")

    if copy_incorrect_dir is not None:
        print(f"Incorrect schemes copied to: {copy_incorrect_dir}")


if __name__ == "__main__":
    main()