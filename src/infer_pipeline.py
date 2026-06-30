import os
import cv2
import numpy as np
import csv
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION PATHS ---
PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
SEG_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_segmentor_best.pt"
ID_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_identity_32_best.pt"
VAL_IMAGES_DIR = PROJECT_ROOT / "data" / "segment_teeth" / "images" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2_polished"

# --- DEFINITIONS ---
ANTERIOR_ORDER = ['CI', 'LI', 'C']
POSTERIOR_TYPES = ['P1', 'P2', 'M1', 'M2', 'M3']

def setup_output_dirs(base_dir):
    """Creates the base output directory and a subfolder for crops."""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(base_dir / "crops", exist_ok=True)
    return base_dir

def get_geometric_quadrant(cx, cy, img_w, img_h):
    """Determines the anatomical quadrant based on centroid position in the OPG."""
    is_upper = cy < (img_h / 2)
    is_patient_right = cx < (img_w / 2) # Left side of image is Patient Right
    
    if is_upper and is_patient_right: return "UR"
    if is_upper and not is_patient_right: return "UL"
    if not is_upper and is_patient_right: return "LR"
    if not is_upper and not is_patient_right: return "LL"

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Determine the coordinates of the intersection rectangle
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Calculate intersection area
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def filter_overlapping_teeth(raw_teeth, iou_threshold=0.45):
    """
    Removes duplicate or highly overlapping masks (artifacts) 
    by keeping the detection with the highest segmentation confidence.
    """
    # Sort teeth by segmentation confidence descending
    raw_teeth.sort(key=lambda x: x['seg_conf'], reverse=True)
    keep_teeth = []
    
    for current_tooth in raw_teeth:
        overlap = False
        for kept_tooth in keep_teeth:
            iou = calculate_iou(current_tooth['box'], kept_tooth['box'])
            if iou > iou_threshold:
                overlap = True
                break
        
        if not overlap:
            keep_teeth.append(current_tooth)
            
    return keep_teeth

def refine_tooth_identities(teeth_data, img_w):
    """Sorts teeth by quadrant and distance to midline to logically assign anteriors."""
    quadrants = {"UR": [], "UL": [], "LR": [], "LL": []}
    for t in teeth_data:
        quadrants[t['quad']].append(t)
        
    refined_teeth = []
    midline = img_w / 2

    for quad, teeth in quadrants.items():
        if not teeth: continue
        
        # Sort teeth by horizontal distance to the midline
        teeth.sort(key=lambda x: abs(x['cx'] - midline))
        
        for i, tooth in enumerate(teeth):
            base_type = tooth['raw_type']
            
            # Positional logic for Anteriors (closest 3 to midline)
            if i == 0:
                final_type = 'CI'
            elif i == 1:
                final_type = 'LI'
            elif i == 2:
                final_type = 'C'
            else:
                # For posteriors, strip any incorrect quadrant prefix the classifier guessed
                final_type = base_type.split("-")[1] if "-" in base_type else base_type
                
                # Fallback: if classifier predicted an anterior type for a posterior position
                if final_type in ANTERIOR_ORDER:
                    final_type = POSTERIOR_TYPES[min(i-3, len(POSTERIOR_TYPES)-1)]

            tooth['final_label'] = f"{quad}-{final_type}"
            refined_teeth.append(tooth)
            
    return refined_teeth

def create_legend_image(teeth_data, width=450, row_height=35):
    """Generates a clean legend panel mapping simple IDs to full Labels."""
    height = max(100, len(teeth_data) * row_height + 80)
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    cv2.putText(legend, "Tooth Identification Legend", (15, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Sort numerically by ID for a readable legend
    teeth_data.sort(key=lambda x: x['id'])
    
    y_offset = 90
    for t in teeth_data:
        text = f"[{t['id']}] : {t['final_label']}  (Conf: {t['id_conf']:.2f})"
        cv2.putText(legend, text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
        y_offset += row_height
        
    return legend

def main():
    print("Loading models...")
    seg_model = YOLO(SEG_MODEL_PATH)
    id_model = YOLO(ID_MODEL_PATH)
    
    out_dir = setup_output_dirs(OUTPUT_DIR)
    
    # Initialize CSV Writer
    csv_file = open(out_dir / "phase2_table.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image', 'Tooth_ID', 'Geometric_Quad', 'Final_Label', 'Classifier_Conf'])
    
    image_paths = list(VAL_IMAGES_DIR.glob("*.jpg")) + list(VAL_IMAGES_DIR.glob("*.png"))
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        print(f"\nProcessing {img_path.name}...")
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        img_h, img_w = img.shape[:2]
        clean_img = img.copy()
        
        # 1. Run Segmentation
        seg_results = seg_model(img, verbose=False)[0]
        
        if seg_results.masks is None:
            print(f"No teeth detected in {img_path.name}.")
            continue
            
        masks = seg_results.masks.data.cpu().numpy()
        boxes = seg_results.boxes.xyxy.cpu().numpy()
        seg_confs = seg_results.boxes.conf.cpu().numpy()
        
        raw_teeth = []
        
        # 2. Extract and Classify Each Tooth (Unordered & Unfiltered)
        for mask, box, s_conf in zip(masks, boxes, seg_confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Get centroid for geometric logic
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            geom_quad = get_geometric_quadrant(cx, cy, img_w, img_h)
            
            # Create masked crop for classification
            mask_resized = cv2.resize(mask, (img_w, img_h))
            tooth_mask = mask_resized[y1:y2, x1:x2]
            crop = img[y1:y2, x1:x2].copy()
            
            # Black out the background leaving only the tooth
            crop[tooth_mask < 0.5] = 0 
            
            # Run Identity Classifier
            id_results = id_model(crop, verbose=False)[0]
            top_class_idx = id_results.probs.top1
            top_class_name = id_model.names[top_class_idx]
            id_conf = float(id_results.probs.top1conf.cpu().numpy())
            
            raw_teeth.append({
                'cx': cx, 'cy': cy,
                'box': (x1, y1, x2, y2),
                'crop': crop,
                'quad': geom_quad,
                'raw_type': top_class_name,
                'id_conf': id_conf,
                'seg_conf': float(s_conf) # Used for IoU filtering
            })

        # 3. Apply IoU Filter to Remove Duplicates/Overlaps
        filtered_teeth = filter_overlapping_teeth(raw_teeth, iou_threshold=0.45)

        # 4. Sort Teeth Geometrically for Sequential Numbering (1, 2, 3...)
        img_cy = img_h / 2
        upper_teeth = [t for t in filtered_teeth if t['cy'] < img_cy]
        lower_teeth = [t for t in filtered_teeth if t['cy'] >= img_cy]
        
        # Sort Upper Teeth: Left of image to Right (Patient Right -> Patient Left)
        upper_teeth.sort(key=lambda t: t['cx'])
        
        # Sort Lower Teeth: Right of image to Left (Patient Left -> Patient Right)
        lower_teeth.sort(key=lambda t: t['cx'], reverse=True)
        
        ordered_teeth = upper_teeth + lower_teeth
        
        # Assign IDs sequentially
        for idx, t in enumerate(ordered_teeth):
            t['id'] = idx + 1

        # 5. Refine Identities via Positional Logic
        teeth_data = refine_tooth_identities(ordered_teeth, img_w)
        
        # 6. Visualization & Saving
        for t in teeth_data:
            x1, y1, x2, y2 = t['box']
            label = t['final_label']
            t_id = t['id']
            
            # Draw simple, clean ID on the main image
            cv2.rectangle(clean_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
            # Add a background rectangle for the text to ensure readability
            cv2.rectangle(clean_img, (x1, y1 - 25), (x1 + 35, y1), (0, 200, 0), -1)
            cv2.putText(clean_img, str(t_id), (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save masked crop into organized folders
            crop_dir = out_dir / "crops" / label
            os.makedirs(crop_dir, exist_ok=True)
            cv2.imwrite(str(crop_dir / f"{img_path.stem}_id{t_id}.png"), t['crop'])
            
            # Write row to CSV
            csv_writer.writerow([img_path.name, t_id, t['quad'], label, round(t['id_conf'], 3)])

        # Save Clean Image
        cv2.imwrite(str(out_dir / f"{img_path.stem}_phase2_clean.png"), clean_img)
        
        # Create and Save Legend
        legend_img = create_legend_image(teeth_data)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_phase2_legend.png"), legend_img)

    csv_file.close()
    print(f"\nPipeline complete! Check outputs at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()