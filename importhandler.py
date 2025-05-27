import ast
import os

def get_top_level_definitions(file_path):
    """Extract top-level functions and classes from a Python file (not inside classes)."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            defs.append(node.name)
    return defs

def generate_imports_and_all(folder_path):
    import_lines = []
    all_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)

            if module_name == "dependencies":
                import_lines.append("from . import dependencies")
                all_names.append("dependencies")
            else:
                names = get_top_level_definitions(file_path)
                if names:
                    import_lines.append(f"from .{module_name} import {', '.join(names)}")
                    all_names.extend(names)

    all_names = sorted(set(all_names))
    return import_lines, all_names

def format_all_block(imported_names):
    lines = ["__all__ = ["]
    for i, name in enumerate(imported_names):
        comma = "," if i < len(imported_names) - 1 else ""
        lines.append(f'    "{name}"{comma}')
    lines.append("]")
    return "\n".join(lines)

def overwrite_init_file(init_file_path):
    folder = os.path.dirname(init_file_path)
    import_lines, all_names = generate_imports_and_all(folder)
    all_block = format_all_block(all_names)

    new_content = "\n".join(import_lines) + "\n\n" + all_block + "\n"

    with open(init_file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

# ✅ Run it
if __name__ == "__main__":
    init_path = r"src\PhysicsOneA\__init__.py"  # Adjust as needed
    overwrite_init_file(init_path)
