import os
import re

# Try to find the current image_folder from annotate.py
current_folder_name = ""
try:
    with open("annotate.py", "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()
        for line in reversed(lines):
            if "image_folder" in line and "=" in line and "batch_15" in line:
                # Extracts "520" from image_folder = r"batch_15\520"
                parts = line.split("\\")
                if len(parts) > 1:
                    current_folder_name = parts[-1].strip().strip('"').strip("'")
                    break
except:
    pass

root = r"batch_15"
folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

grand_total_images = 0
active_total_images = 0
active_total_annotated = 0

print(f"{'Folder':<10} | {'Images':<10} | {'Annotated':<10} | {'Progress':<10}")
print("-" * 50)

for folder in folders:
    folder_path = os.path.join(root, folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    
    labels_dir = os.path.join(folder_path, "labels")
    if os.path.exists(labels_dir):
        annotated = [f for f in os.listdir(labels_dir) if f.lower().endswith(".json")]
    else:
        annotated = []
    
    img_count = len(images)
    ann_count = len(annotated)
    
    grand_total_images += img_count
    
    # Show folder if it has annotations OR if it is the current folder
    if folder == current_folder_name or ann_count > 0:
        active_total_images += img_count
        active_total_annotated += ann_count
        
        pct = (ann_count / img_count * 100) if img_count > 0 else 0
        marker = " <--" if folder == current_folder_name else ""
        print(f"{folder:<10} | {img_count:<10} | {ann_count:<10} | {pct:>8.1f}%{marker}")

print("-" * 50)
if active_total_images > 0:
    active_pct = (active_total_annotated / active_total_images * 100)
    print(f"{'TOTAL':<10} | {active_total_images:<10} | {active_total_annotated:<10} | {active_pct:>8.1f}% (Active)")
    
    grand_pct = (active_total_annotated / grand_total_images * 100)
    print(f"{'BATCH':<10} | {grand_total_images:<10} | {active_total_annotated:<10} | {grand_pct:>8.1f}% (Full Batch)")
else:
    print("No active folders found.")
