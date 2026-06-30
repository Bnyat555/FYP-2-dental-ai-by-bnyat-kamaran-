import json
from pathlib import Path

json_path = Path(r"C:\Users\MSI\Desktop\FYP AI Data Set\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("CATEGORIES:")
for cat in data.get("categories", []):
    print(cat)

print("\nFIRST 5 ANNOTATIONS:")
for ann in data.get("annotations", [])[:5]:
    print(ann)