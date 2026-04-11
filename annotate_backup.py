import cv2
import json
import os
import numpy as np

# Define 16 facial region classes with distinct BGR colors
classes = {
    # --- Frontal Regions ---
    "Left Side Frontal": (0, 255, 0),          # 1 (Green)
    "Right Side Frontal": (34, 139, 34),       # 2 (Forest Green)
    "Central Frontal region": (255, 0, 0),     # 3 (Blue)
    
    # --- Eye Regions ---
    "Upper eyelid": (0, 255, 255),             # 4 (Yellow)
    
    # --- Canthus (Split Left/Right) ---
    "Left Lateral Canthus": (255, 0, 255),     # 5 (Magenta)
    "Right Lateral Canthus": (147, 20, 255),   # t (Deep Pink/Purple)
    "Left Medial Canthus": (255, 165, 0),      # 6 (Cyan/Orange mix)
    "Right Medial Canthus": (218, 165, 32),    # y (Goldenrod)
    
    # --- Mouth/Chin/Nose ---
    "Mental region(chin)": (128, 0, 128),      # 7 (Purple)
    "Border(lip region)": (0, 0, 255),         # 8 (Red)
    "Upper lip": (255, 255, 0),                # 9 (Cyan)
    "Nasal region": (203, 182, 255),           # 0 (Pinkish)
    
    # --- Cheeks (Split) ---
    "Upper Cheek": (71, 99, 255),              # u (Tomato)
    "Lower Cheek": (100, 149, 237),            # l (Cornflower Blue)
    
    # --- Ears (Split) ---
    "Ear Pinna": (128, 128, 0),                # p (Teal)
    "Ear Lobule": (0, 128, 128)                # b (Olive)
}

# Mapping keys to class names
key_map = {
    ord('1'): "Left Side Frontal",
    ord('2'): "Right Side Frontal",
    ord('3'): "Central Frontal region",
    ord('4'): "Upper eyelid",
    
    # Canthus Keys
    ord('5'): "Left Lateral Canthus",
    ord('t'): "Right Lateral Canthus", 
    ord('6'): "Left Medial Canthus",
    ord('y'): "Right Medial Canthus", 
    
    ord('7'): "Mental region(chin)",
    ord('8'): "Border(lip region)",
    ord('9'): "Upper lip",
    ord('0'): "Nasal region",
    
    # Letters
    ord('u'): "Upper Cheek",
    ord('l'): "Lower Cheek",
    ord('p'): "Ear Pinna",
    ord('b'): "Ear Lobule"
}

# Global variables
drawing = False
current_class = "Left Side Frontal"
annotations = []
points = []
undo_stack = []
fixed_size = (800, 800) 

def resize_image(image, size=fixed_size):
    return cv2.resize(image, size)

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def save_annotations(image_path, scale_x, scale_y):
    # Identify the base directory and filename
    folder_path = os.path.dirname(image_path)
    filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    # --- 1. Save JSON in 'labels' folder ---
    labels_dir = os.path.join(folder_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    json_path = os.path.join(labels_dir, filename_no_ext + ".json")
    
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"✅ JSON saved to: {json_path}")

    # --- 2. Save Image Preview in 'annotated_images' folder ---
    previews_dir = os.path.join(folder_path, "annotated_images")
    os.makedirs(previews_dir, exist_ok=True)
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("❌ Could not reload original image for saving annotated image.")
        return

    # Draw polygons on the preview image
    for ann in annotations:
        poly = np.array(ann["polygon"], dtype=np.float32)
        poly[:, 0] *= scale_x
        poly[:, 1] *= scale_y
        poly = poly.astype(np.int32)
        color = classes[ann["class"]]
        cv2.polylines(original_img, [poly], isClosed=True, color=color, thickness=2)
        cv2.putText(original_img, ann["class"], tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    output_path = os.path.join(previews_dir, filename_no_ext + "_annotated.jpg")
    cv2.imwrite(output_path, original_img)
    print(f"✅ Preview saved to: {output_path}")

def load_annotations(image_path):
    # Try to load JSON from the 'labels' folder
    folder_path = os.path.dirname(image_path)
    filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(folder_path, "labels", filename_no_ext + ".json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return []

def annotate_image(image_path):
    global img, annotations, undo_stack, points, current_class

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error loading image: {image_path}")
        return

    original_h, original_w = original_img.shape[:2]
    scale_x = original_w / fixed_size[0]
    scale_y = original_h / fixed_size[1]

    img = resize_image(original_img)
    annotations.clear()
    annotations.extend(load_annotations(image_path))
    undo_stack.clear()
    points.clear()

    cv2.namedWindow("Polygon Annotation Tool")
    cv2.setMouseCallback("Polygon Annotation Tool", mouse_callback)

    while True:
        temp_img = img.copy()

        # Draw saved polygons
        for ann in annotations:
            poly = ann["polygon"]
            color = classes[ann["class"]]
            cv2.polylines(temp_img, [np.array(poly, dtype=np.int32)], isClosed=True, color=color, thickness=2)

        # Draw in-progress polygon
        if len(points) > 1:
            cv2.polylines(temp_img, [np.array(points, dtype=np.int32)], isClosed=False, color=classes[current_class], thickness=1)
        for p in points:
            cv2.circle(temp_img, p, 3, classes[current_class], -1)

        # UI Overlay - Instructions
        cv2.putText(temp_img, f"Class: {current_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classes[current_class], 2)
        
        # Row 1
        cv2.putText(temp_img, "1:L-Front 2:R-Front 3:C-Front 4:Up-Eye 7:Chin 8:Lip-B 9:Up-Lip 0:Nose", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        # Row 2 (Eye Corners)
        cv2.putText(temp_img, "5:L-Lat-Can t:R-Lat-Can | 6:L-Med-Can y:R-Med-Can", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        # Row 3 (Cheeks/Ears/Controls)
        cv2.putText(temp_img, "u:Up-Ch l:Low-Ch p:Pinna b:Lobule | s:Save z:Undo x:Redo q:Next", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        cv2.imshow("Polygon Annotation Tool", temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            save_annotations(image_path, scale_x, scale_y)
            break
        elif key == ord("q"):
            break 
        elif key == ord("z") and annotations:
            undo_stack.append(annotations.pop())
            print("Undo last annotation")
        elif key == ord("x") and undo_stack: 
            annotations.append(undo_stack.pop())
            print("Redo annotation")
        elif key == ord("r"):
            points.clear()
            print("Reset polygon")
        elif key in [13, 10] and len(points) >= 3:  # Enter
            annotations.append({"class": current_class, "polygon": points.copy()})
            print(f"✅ Saved polygon: {current_class}")
            points.clear()
        elif key in key_map:
            current_class = key_map[key]
            print(f"Selected Class: {current_class}")

    cv2.destroyAllWindows()

def annotate_folder(folder_path):
    # Ensure folder path is valid
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return

    # Filter for image files
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    total = len(image_files)
    print(f"Found {total} images in {folder_path}")

    # Check which images already have labels
    labels_dir = os.path.join(folder_path, "labels")
    for i, filename in enumerate(image_files, 1):
        filename_no_ext = os.path.splitext(filename)[0]
        json_path = os.path.join(labels_dir, filename_no_ext + ".json")
        if os.path.exists(json_path):
            print(f"⏭️  [{i}/{total}] Already annotated: {filename}. Skipping.")
            continue
        print(f"\n🔍 [{i}/{total}] Annotating: {filename}")
        annotate_image(os.path.join(folder_path, filename))


def annotate_all_folders(base_path):
    """Iterate through all subfolders in base_path, one by one."""
    if not os.path.exists(base_path):
        print(f"❌ Error: Base folder not found at {base_path}")
        return

    # Get all subfolders (sorted numerically)
    subfolders = sorted(
        [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))],
        key=lambda x: int(x) if x.isdigit() else x
    )
    total_folders = len(subfolders)
    print(f"📂 Found {total_folders} folders in {base_path}")
    print("Controls: 's' = save & next image | 'q' = skip image | 'n' in console = skip folder | Ctrl+C = quit\n")

    for i, folder_name in enumerate(subfolders, 1):
        folder_full = os.path.join(base_path, folder_name)

        # Skip folders that already have a 'labels' subfolder (already annotated)
        labels_dir = os.path.join(folder_full, "labels")
        image_files = [f for f in os.listdir(folder_full) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if os.path.exists(labels_dir):
            existing_labels = len([f for f in os.listdir(labels_dir) if f.endswith(".json")])
            if existing_labels >= len(image_files):
                print(f"⏭️  [{i}/{total_folders}] Folder '{folder_name}' already fully annotated ({existing_labels} labels). Skipping.")
                continue

        print(f"\n{'='*60}")
        print(f"📂 [{i}/{total_folders}] Starting folder: {folder_name}")
        print(f"{'='*60}")
        annotate_folder(folder_full)
        print(f"✅ Done with folder: {folder_name}")


# 📂 UPDATE BASE PATH HERE
base_folder = r"batch_15"
annotate_all_folders(base_folder)