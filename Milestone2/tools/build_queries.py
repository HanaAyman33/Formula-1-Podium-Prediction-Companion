"""Assemble queries.txt from queries/q1.cypher ... q5.cypher.

Each file must start with its own header line, exactly "//N. Query N". The output is written as
UTF-8 without a BOM, with LF line endings and one blank line between queries, so the grader's
parser sees the same text on every operating system.

Usage: python tools/build_queries.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file is in tools/)
N_QUERIES = 5


def main():
    parts, errors = [], []
    for n in range(1, N_QUERIES + 1):
        path = os.path.join(HERE, "queries", f"q{n}.cypher")
        if not os.path.exists(path):
            errors.append(f"missing {path}")
            continue
        # utf-8-sig silently drops a BOM if an editor added one
        with open(path, encoding="utf-8-sig") as f:
            text = f.read().replace("\r\n", "\n").strip()
        header = text.splitlines()[0] if text else ""
        if header != f"//{n}. Query {n}":
            errors.append(f"q{n}.cypher: first line is {header!r}, expected '//{n}. Query {n}'")
        body = "\n".join(text.splitlines()[1:]).strip()
        if not body:
            errors.append(f"q{n}.cypher: no query after the header")
        if body.rstrip().endswith(";"):
            errors.append(f"q{n}.cypher: remove the trailing ';' (the other queries have none)")
        if re.search(r"//\d+\. Query \d+", body):
            errors.append(f"q{n}.cypher: contains a second header line")
        parts.append(text)

    if errors:
        print("queries.txt NOT written:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    out = os.path.join(HERE, "queries.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"wrote {out} ({N_QUERIES} queries, UTF-8, no BOM)")


if __name__ == "__main__":
    main()
