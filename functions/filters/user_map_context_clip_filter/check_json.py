#!/usr/bin/env python3
import argparse
import json
import sys


def line_col(text, idx):
    line = text.count("\n", 0, idx) + 1
    col = idx - text.rfind("\n", 0, idx)
    return line, col


def find_issues(text):
    issues = []
    for i, ch in enumerate(text):
        if ch == "\t":
            line, col = line_col(text, i)
            issues.append(f"TAB at {line}:{col}")
        elif ord(ch) > 0x7E:
            line, col = line_col(text, i)
            issues.append(
                f"NON-ASCII U+{ord(ch):04X} {repr(ch)} at {line}:{col}"
            )
    return issues


def read_text(path):
    if path == "-" or path is None:
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSON and flag tabs or non-ASCII characters."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="JSON file path, or '-' to read from stdin",
    )
    args = parser.parse_args()

    text = read_text(args.path)

    ok = True
    try:
        json.loads(text)
        print("JSON OK")
    except json.JSONDecodeError as exc:
        print(f"JSON ERROR: {exc}")
        ok = False

    issues = find_issues(text)
    for issue in issues:
        print(issue)
    if issues:
        ok = False

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
