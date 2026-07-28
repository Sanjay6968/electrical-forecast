import os
import ast

def remove_docstrings(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
    for child in ast.iter_child_nodes(node):
        remove_docstrings(child)

def strip_comments_and_docstrings(source):
    try:
        parsed = ast.parse(source)
        remove_docstrings(parsed)
        return ast.unparse(parsed)
    except Exception as e:
        print(f"Failed to parse source: {e}")
        return source

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py") and file != "strip.py":
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            cleaned = strip_comments_and_docstrings(source)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
