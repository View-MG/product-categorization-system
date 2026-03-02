import json

notebook_path = "product-categorization-system.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "def basic_clean(" in source:
            # We want to insert the remap logic just like we did in `src/data/prepare.py`
            old_code = '    df["label_coarse"] = df["label_coarse"].astype(str).str.strip()\n'
            new_code = '''    df["label_coarse"] = df["label_coarse"].astype(str).str.strip()

    def remap_label(lbl: str) -> str:
        lbl = str(lbl).lower()
        if lbl in ["snacks", "snack"]: return "snack"
        if lbl in ["beverages", "beverage"]: return "beverage"
        return lbl
        
    df["label_coarse"] = df["label_coarse"].apply(remap_label)
'''
            if old_code in source and "remap_label" not in source:
                new_source = source.replace(old_code, new_code)
                # re-split into list of strings with newlines
                lines = [line + "\\n" for line in new_source.split("\\n")[:-1]] + [new_source.split("\\n")[-1]]
                cell["source"] = lines

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched")
