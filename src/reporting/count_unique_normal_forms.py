from pathlib import Path
import argparse
import hashlib


def normalize_scheme_file(file_path):
    """
    Read a normal form file and normalize its contents.

    We ignore:
    - empty lines
    - comment lines starting with #

    We also strip extra spaces so that:
        1001 1001 0001
    and:
        1001   1001   0001
    are treated the same.
    """

    normalized_lines = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Bad line in {file_path}:\n{line}\n"
                    "Expected exactly 3 bit vectors per row."
                )

            normalized_line = " ".join(parts)
            normalized_lines.append(normalized_line)

    return "\n".join(normalized_lines)


def hash_scheme(normalized_text):
    """
    Create a short fingerprint/hash for a scheme.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def count_unique_normal_forms(directory, extension="txt", report_file=None):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    if not extension.startswith("."):
        extension = "." + extension

    files = sorted(directory.glob(f"*{extension}"))

    if not files:
        print(f"No {extension} files found in {directory}")
        return

    groups = {}

    for file_path in files:
        normalized_text = normalize_scheme_file(file_path)
        scheme_hash = hash_scheme(normalized_text)

        if scheme_hash not in groups:
            groups[scheme_hash] = {
                "text": normalized_text,
                "files": []
            }

        groups[scheme_hash]["files"].append(file_path.name)

    total_files = len(files)
    unique_count = len(groups)

    lines = []

    lines.append(f"Directory checked: {directory}")
    lines.append(f"Total files checked: {total_files}")
    lines.append(f"Unique normal forms: {unique_count}")
    lines.append("")

    for group_number, (scheme_hash, data) in enumerate(groups.items(), start=1):
        files_in_group = data["files"]

        lines.append("=" * 60)
        lines.append(f"Unique scheme group {group_number}")
        lines.append(f"Hash: {scheme_hash[:16]}")
        lines.append(f"Number of files in this group: {len(files_in_group)}")
        lines.append("Files:")

        for filename in files_in_group:
            lines.append(f"  - {filename}")

        lines.append("")
        lines.append("Representative normal form:")
        lines.append(data["text"])
        lines.append("")

    output = "\n".join(lines)

    print(output)

    if report_file is not None:
        with open(report_file, "w") as f:
            f.write(output)

        print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count unique normal form scheme files in a directory."
    )

    parser.add_argument(
        "directory",
        help="Folder containing normal form .txt files."
    )

    parser.add_argument(
        "--extension",
        default="txt",
        help="File extension to check. Default is txt."
    )

    parser.add_argument(
        "--report",
        help="Optional output report file, for example unique_report.txt."
    )

    args = parser.parse_args()

    count_unique_normal_forms(
        directory=args.directory,
        extension=args.extension,
        report_file=args.report
    )