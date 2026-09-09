"""Generic script -> notebook assembler for this project's convention:
scratch-file-per-cell + ast.parse gate on every cell, assembled into one .ipynb.

Splits a .py source file into cells at top-level statement boundaries (each
top-level def/class is its own cell; consecutive simple statements are grouped;
the `if __name__ == "__main__":` guard is stripped and its body becomes the
final cell(s), unindented). Every cell is ast.parse-validated before being added.
"""
import ast
import sys
import nbformat as nbf


def split_into_cells(source: str):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    cells = []
    buffer = []

    def flush():
        if buffer:
            text = "".join(buffer).strip("\n")
            if text.strip():
                cells.append(text)
            buffer.clear()

    for node in tree.body:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        chunk_lines = lines[start:end]

        if isinstance(node, ast.If) and _is_main_guard(node):
            flush()
            # unindent the body of `if __name__ == "__main__":`
            body_start = node.body[0].lineno - 1
            body_end = node.body[-1].end_lineno
            body_lines = lines[body_start:body_end]
            dedented = _dedent(body_lines)
            cells.append(dedented.strip("\n"))
            continue

        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            flush()
            cells.append("".join(chunk_lines).strip("\n"))
            continue

        buffer.extend(chunk_lines)

    flush()
    for c in cells:
        ast.parse(c)  # gate: every cell must independently parse
    return cells


def _is_main_guard(node: ast.If) -> bool:
    try:
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and isinstance(test.ops[0], ast.Eq)
        )
    except Exception:
        return False


def _dedent(chunk_lines):
    non_empty = [l for l in chunk_lines if l.strip()]
    if not non_empty:
        return "".join(chunk_lines)
    indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    return "".join(l[indent:] if len(l) >= indent else l for l in chunk_lines)


def build_notebook(py_path: str, ipynb_path: str, title_md: str):
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()
    cells_src = split_into_cells(source)

    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(title_md))
    for c in cells_src:
        nb.cells.append(nbf.v4.new_code_cell(c))

    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"assembled {len(cells_src)} code cells -> {ipynb_path}")


if __name__ == "__main__":
    py_path, ipynb_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
    build_notebook(py_path, ipynb_path, title)
