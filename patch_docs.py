import re

def patch_file(path_str):
    with open(path_str, "r") as f:
        content = f.read()

    # Replacements
    content = content.replace("beverages", "beverage")
    content = content.replace("snacks", "snack")
    content = content.replace("dry_food", "")
    content = content.replace("non_food", "")
    content = content.replace("beverage/snack//", "beverage/snack")
    content = content.replace("beverage, snack, , ", "beverage, snack")
    content = content.replace("beverage, snack", "beverage, snack")
    content = content.replace("- `beverage`\n- `snack`\n- ``\n- ``", "- `beverage`\n- `snack`")
    content = content.replace("beverage/snack", "beverage/snack")
    
    # 4 classes to 2 classes
    content = content.replace("4 classes", "2 classes")
    content = content.replace("Number of classes  | 4", "Number of classes  | 2")
    content = content.replace("values 0–3", "values 0–1")
    content = content.replace("Output Logits: (B, 4)   ← pair with nn.CrossEntropyLoss", "Output Logits: (B, 2)   ← pair with nn.CrossEntropyLoss")
    content = content.replace("Linear(1280 → 4)", "Linear(1280 → 2)")
    content = content.replace("— [beverage, snack, , ]", "— [beverage, snack]")
    content = content.replace("scalar class index (0–3)", "scalar class index (0–1)")

    # TRAIN.md specific
    content = content.replace("  /0000000000789.jpg\n", "")
    content = content.replace("  /0000000000999.jpg\n", "")
    
    # FEATURE_ENGINEERING.md specific
    content = content.replace('  "beverage": 0,\n  "snack": 1,\n  "": 2,\n  "": 3', '  "beverage": 0,\n  "snack": 1')

    # Remove lingering empty lines or broken bullets
    content = re.sub(r"- ``\n", "", content)
    content = re.sub(r'  "": \d,\n', "", content)
    content = re.sub(r'  "": \d\n', "\n", content)

    with open(path_str, "w") as f:
        f.write(content)

patch_file("TRAIN.md")
patch_file("FEATURE_ENGINEERING.md")
print("Done.")
# Patch notebook remaining docs
content = open("product-categorization-system.ipynb").read()
content = content.replace("**Categories:** `beverages`, `snacks`, `dry_food`, `non_food`", "**Categories:** `beverage`, `snack`")
with open("product-categorization-system.ipynb", "w") as f: f.write(content)
print("Notebook doc patched")
