"""Simple fixed-width table printer for CLI reports."""

from __future__ import annotations


def print_table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None) -> None:
    if not headers:
        return
    if widths is None:
        widths = []
        for col_idx, header in enumerate(headers):
            w = len(header)
            for row in rows:
                if col_idx < len(row):
                    w = max(w, len(str(row[col_idx])))
            widths.append(w + 2)

    header_line = ''.join(f'{h:<{widths[i]}}' for i, h in enumerate(headers))
    print(header_line)
    print('-' * sum(widths))
    for row in rows:
        cells = []
        for i, w in enumerate(widths):
            val = row[i] if i < len(row) else ''
            cells.append(f'{val:<{w}}')
        print(''.join(cells))
