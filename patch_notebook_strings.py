import json

notebook_path = "product-categorization-system.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        for i, line in enumerate(source):
            if line == 'print(f"\\n\n':
                source[i] = 'print(f"\\n" \\\n'
            elif line == 'Output directory for this run: {run_dir}\\n\n':
                source[i] = '      f"Output directory for this run: {run_dir}\\n")\n'
            elif line == '")\n' and i > 0 and 'run_dir' in source[i-1]:
                source[i] = '' # Remove the lone closing parenthesis line
                
            elif line == 'Training complete! Results saved in {run_dir}")\n':
                source[i] = '      f"Training complete! Results saved in {run_dir}")\n'

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook strings patched")
