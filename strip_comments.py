import os
import ast

def strip_comments_and_docstrings(source):
    try:
        # Parse into AST
        parsed = ast.parse(source)
        # We can use ast.unparse to regenerate code without comments/docstrings in Python 3.9+
        # But to just remove docstrings and keep formatting mostly, it's better to just use ast.unparse
        # Let's remove docstrings explicitly from modules, classes, and functions
        for node in ast.walk(parsed):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                continue
            if not len(node.body):
                continue
            if not isinstance(node.body[0], ast.Expr):
                continue
            if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                continue
            # Remove the docstring node
            node.body = node.body[1:]
            
        return ast.unparse(parsed)
    except Exception as e:
        print(f"Failed to parse source: {e}")
        return source

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "strip_comments.py":
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                
                cleaned = strip_comments_and_docstrings(source)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)

if __name__ == "__main__":
    process_directory(".")
