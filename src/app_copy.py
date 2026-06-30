import hashlib
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from pathology_utils import detect_pathology, draw_pathology_boxes, summarize_pathology


st.set_page_config(
    page_title="Dental-AI Workstation",
    layout="wide",
    page_icon="🦷",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }
        .stApp {
            background: #f4f7fb;
        }
        .block-container {
            max-width: 94%;
            padding-top: 1.2rem;
            padding-bottom: 2.0rem;
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }

        .app-hero {
            background: linear-gradient(180deg, #0b1730 0%, #0d1c39 100%);
            border: 1px solid #142847;
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }
        .app-title {
            color: #f8fafc;
            font-size: 2.0rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
        }
        .app-subtitle {
            color: #b8c6dc;
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }
        .case-card {
            background: linear-gradient(180deg, #0b1730 0%, #0d1c39 100%);
            border: 1px solid #142847;
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }
        .case-card-inner {
            border: 1px solid rgba(255, 255, 255, 0.10);
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 0.8rem 1rem;
        }
        .case-label {
            color: #8ea6c7;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .case-value {
            color: #ffffff;
            font-size: 1.25rem;
            font-weight: 800;
        }

        .ui-card {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .ui-card-title {
            color: #142033;
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .ui-card-subtitle {
            color: #607289;
            font-size: 0.92rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 0.95rem;
        }
        .summary-card {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }
        .summary-label {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .summary-value {
            color: #142033;
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.0;
        }

        .viewer-shell {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 22px;
            padding: 0.8rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .findings-shell {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-top: 1rem;
        }
        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }
        .uploaded-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
        }
        .uploaded-name {
            font-weight: 700;
            color: #142033;
            word-break: break-word;
        }
        .uploaded-meta {
            color: #64748b;
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }
        .mode-chip {
            display: inline-block;
            background: #eef6ff;
            border: 1px solid #cfe5ff;
            color: #0f4c81;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.55rem;
        }

        .stButton > button, .stDownloadButton > button {
            width: 100%;
            border: 0;
            border-radius: 14px;
            padding: 0.76rem 1rem;
            font-weight: 700;
            background: #1fa2e1;
            color: white;
            box-shadow: 0 8px 18px rgba(31, 162, 225, 0.22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #1188c4;
        }

        [data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 16px;
            padding: 0.35rem 0.55rem 0.55rem 0.55rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stFileUploader"] section {
            border: 0;
            background: transparent;
        }

        .section-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 0.4rem 0 0.9rem 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #eef3f8;
            border-radius: 10px;
            padding: 0.35rem 0.8rem;
        }
        .stTabs [aria-selected="true"] {
            background: #dcecff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
SEG_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_segmentor_best.pt"
ID_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_identity_32_best.pt"

ANTERIOR_ORDER = ["CI", "LI", "C"]
POSTERIOR_TYPES = ["P1", "P2", "M1", "M2", "M3"]
TEAL_BGR = (181, 138, 42)
AMBER_BGR = (17, 158, 237)
WHITE_BGR = (248, 250, 252)
SLATE_BGR = (41, 55, 75)
LIGHT_CARD_BGR = (250, 252, 255)
DARK_TEXT_BGR = (45, 58, 76)


@st.cache_resource
def load_models():
    return YOLO(SEG_MODEL_PATH), YOLO(ID_MODEL_PATH)


try:
    seg_model, id_model = load_models()
except Exception:
    st.error(f"Failed to load models. Check these files: {SEG_MODEL_PATH} and {ID_MODEL_PATH}")
    st.stop()


if "case_id" not in st.session_state:
    st.session_state.case_id = f"#{str(uuid.uuid4())[:4].upper()}-{str(uuid.uuid4())[:1].upper()}"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = f"upload_{uuid.uuid4().hex}"
if "image_sig" not in st.session_state:
    st.session_state.image_sig = None
if "raw_img" not in st.session_state:
    st.session_state.raw_img = None
if "teeth_data" not in st.session_state:
    st.session_state.teeth_data = []
if "infer_time" not in st.session_state:
    st.session_state.infer_time = 0.0
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_size" not in st.session_state:
    st.session_state.uploaded_size = 0
if "render_png" not in st.session_state:
    st.session_state.render_png = None
if "pathology_data" not in st.session_state:
    st.session_state.pathology_data = []


def reset_session():
    st.session_state.case_id = f"#{str(uuid.uuid4())[:4].upper()}-{str(uuid.uuid4())[:1].upper()}"
    st.session_state.uploader_key = f"upload_{uuid.uuid4().hex}"
    st.session_state.image_sig = None
    st.session_state.raw_img = None
    st.session_state.teeth_data = []
    st.session_state.infer_time = 0.0
    st.session_state.uploaded_name = None
    st.session_state.uploaded_size = 0
    st.session_state.render_png = None
    st.session_state.pathology_data = []


def human_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def get_geometric_quadrant(cx, cy, img_w, img_h):
    is_upper = cy < (img_h / 2)
    is_patient_right = cx < (img_w / 2)
    if is_upper and is_patient_right:
        return "UR"
    if is_upper and not is_patient_right:
        return "UL"
    if not is_upper and is_patient_right:
        return "LR"
    return "LL"


def get_box_quadrant(box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return get_geometric_quadrant(cx, cy, img_w, img_h)


def build_pathology_df(pathology_data, img_w, img_h):
    rows = []
    for idx, det in enumerate(sorted(pathology_data, key=lambda d: d["conf"], reverse=True), start=1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            {
                "ID": idx,
                "Finding": det["label"],
                "Quad": get_box_quadrant(det["bbox"], img_w, img_h),
                "Confidence": round(det["conf"], 2),
                "Box": f"[{x1}, {y1}, {x2}, {y2}]",
            }
        )
    return pd.DataFrame(rows)


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (
        (x2_1 - x1_1) * (y2_1 - y1_1)
        + (x2_2 - x1_2) * (y2_2 - y1_2)
        - inter_area
    )
    return inter_area / union_area if union_area > 0 else 0.0


def filter_overlapping_teeth(raw_teeth, iou_threshold=0.45):
    raw_teeth.sort(key=lambda x: x["seg_conf"], reverse=True)
    keep_teeth = []
    for current_tooth in raw_teeth:
        has_overlap = any(
            calculate_iou(current_tooth["box"], kept["box"]) > iou_threshold
            for kept in keep_teeth
        )
        if not has_overlap:
            keep_teeth.append(current_tooth)
    return keep_teeth


def refine_tooth_identities(teeth_data, img_w):
    quadrants = {"UR": [], "UL": [], "LR": [], "LL": []}
    for tooth in teeth_data:
        quadrants[tooth["quad"]].append(tooth)

    refined_teeth = []
    midline = img_w / 2

    for quad, teeth in quadrants.items():
        if not teeth:
            continue
        teeth.sort(key=lambda x: abs(x["cx"] - midline))
        for i, tooth in enumerate(teeth):
            base_type = tooth["raw_type"]
            if i == 0:
                final_type = "CI"
            elif i == 1:
                final_type = "LI"
            elif i == 2:
                final_type = "C"
            else:
                final_type = base_type.split("-")[1] if "-" in base_type else base_type
                if final_type in ANTERIOR_ORDER:
                    final_type = POSTERIOR_TYPES[min(i - 3, len(POSTERIOR_TYPES) - 1)]
            tooth["final_type"] = final_type
            tooth["final_label"] = f"{quad}-{final_type}"
            refined_teeth.append(tooth)
    return refined_teeth


def process_image(raw_img):
    start_time = time.time()
    img_h, img_w = raw_img.shape[:2]

    seg_results = seg_model(raw_img, verbose=False)[0]
    if seg_results.masks is None:
        return [], 0.0

    masks = seg_results.masks.data.cpu().numpy()
    boxes = seg_results.boxes.xyxy.cpu().numpy()
    seg_confs = seg_results.boxes.conf.cpu().numpy()

    raw_teeth = []
    for mask, box, seg_conf in zip(masks, boxes, seg_confs):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        geom_quad = get_geometric_quadrant(cx, cy, img_w, img_h)

        mask_resized = cv2.resize(mask, (img_w, img_h))
        tooth_mask = mask_resized[y1:y2, x1:x2]
        crop = raw_img[y1:y2, x1:x2].copy()
        crop[tooth_mask < 0.5] = 0

        id_results = id_model(crop, verbose=False)[0]
        raw_teeth.append(
            {
                "cx": cx,
                "cy": cy,
                "box": (x1, y1, x2, y2),
                "quad": geom_quad,
                "raw_type": id_model.names[id_results.probs.top1],
                "id_conf": float(id_results.probs.top1conf.cpu().numpy()),
                "seg_conf": float(seg_conf),
                "mask": mask_resized,
            }
        )

    filtered_teeth = filter_overlapping_teeth(raw_teeth)
    img_cy = img_h / 2
    upper_teeth = sorted(
        [tooth for tooth in filtered_teeth if tooth["cy"] < img_cy],
        key=lambda tooth: tooth["cx"],
    )
    lower_teeth = sorted(
        [tooth for tooth in filtered_teeth if tooth["cy"] >= img_cy],
        key=lambda tooth: tooth["cx"],
        reverse=True,
    )

    ordered_teeth = upper_teeth + lower_teeth
    for idx, tooth in enumerate(ordered_teeth, start=1):
        tooth["id"] = idx

    infer_time = time.time() - start_time
    return refine_tooth_identities(ordered_teeth, img_w), infer_time


def build_findings_df(teeth_data, conf_thresh):
    rows = []
    for tooth in sorted(teeth_data, key=lambda t: t["id"]):
        rows.append(
            {
                "ID": tooth["id"],
                "Label": tooth["final_label"],
                "Quad": tooth["quad"],
                "Type": tooth["final_type"],
                "ID Conf": round(tooth["id_conf"], 2),
                "Seg Conf": round(tooth["seg_conf"], 2),
                "Status": "Review" if tooth["id_conf"] < conf_thresh else "OK",
            }
        )
    return pd.DataFrame(rows)


def quadrant_counts(teeth_data):
    return {quad: sum(1 for tooth in teeth_data if tooth["quad"] == quad) for quad in ["UR", "UL", "LL", "LR"]}


def draw_overlay_card(img, x, y, width, title, lines, accent=(31, 162, 225), alpha=0.84):
    line_gap = 18
    title_gap = 20
    pad_x = 12
    height = 16 + title_gap + max(len(lines), 1) * line_gap + 10

    x = max(0, min(x, img.shape[1] - width - 2))
    y = max(0, min(y, img.shape[0] - height - 2))

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), LIGHT_CARD_BGR, -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + 4), accent, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + height), (213, 223, 235), 1)

    cv2.putText(
        img,
        title,
        (x + pad_x, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        DARK_TEXT_BGR,
        1,
        cv2.LINE_AA,
    )

    cursor_y = y + 24 + title_gap
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + pad_x, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            DARK_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
        cursor_y += line_gap


def build_map_lines(teeth_data, conf_thresh, view_mode):
    counts = quadrant_counts(teeth_data)
    review_counts = {
        quad: sum(1 for tooth in teeth_data if tooth["quad"] == quad and tooth["id_conf"] < conf_thresh)
        for quad in ["UR", "UL", "LL", "LR"]
    }
    if view_mode == "Presentation Mode":
        return [
            f"UR  {counts['UR']} teeth   {review_counts['UR']} review",
            f"UL  {counts['UL']} teeth   {review_counts['UL']} review",
            f"LL  {counts['LL']} teeth   {review_counts['LL']} review",
            f"LR  {counts['LR']} teeth   {review_counts['LR']} review",
        ]

    lines = []
    for quad in ["UR", "UL", "LL", "LR"]:
        quad_teeth = sorted([t for t in teeth_data if t["quad"] == quad], key=lambda t: t["id"])
        short = " ".join(t["final_type"] for t in quad_teeth[:6]) if quad_teeth else "--"
        lines.append(f"{quad}  {counts[quad]} teeth  |  {short}")
    return lines


def encode_png(img_bgr):
    ok, buffer = cv2.imencode(".png", img_bgr)
    return buffer.tobytes() if ok else None


def render_image(
    raw_img,
    teeth_data,
    case_id,
    infer_time,
    selected_quadrant,
    view_mode,
    show_masks,
    show_boxes,
    show_full_labels,
    mask_opacity,
    conf_thresh,
):
    display_img = raw_img.copy()
    img_h, img_w = display_img.shape[:2]

    if selected_quadrant != "All":
        display_img = cv2.convertScaleAbs(display_img, alpha=0.42, beta=0)

    mask_overlay = np.zeros_like(display_img, dtype=np.uint8)
    low_conf_count = sum(1 for tooth in teeth_data if tooth["id_conf"] < conf_thresh)

    for tooth in teeth_data:
        quad = tooth["quad"]
        if selected_quadrant != "All" and quad != selected_quadrant:
            continue

        x1, y1, x2, y2 = tooth["box"]
        is_low = tooth["id_conf"] < conf_thresh
        color = AMBER_BGR if is_low else TEAL_BGR
        mask_binary = (tooth["mask"] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if show_masks:
            mask_overlay[mask_binary > 0] = color
        cv2.drawContours(display_img, contours, -1, color, 1 if view_mode == "Presentation Mode" else 2)

        if show_boxes:
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)

        if view_mode == "Presentation Mode":
            label = str(tooth["id"])
            font_scale = 0.34
        else:
            label = tooth["final_label"] if show_full_labels else str(tooth["id"])
            font_scale = 0.36 if len(label) < 9 else 0.31

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_x = max(4, min(x1, img_w - tw - 8))
        label_y = max(th + 8, y1 - 6)
        cv2.rectangle(
            display_img,
            (label_x - 3, label_y - th - 5),
            (label_x + tw + 5, label_y + 3),
            (32, 38, 47),
            -1,
        )
        cv2.putText(
            display_img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            WHITE_BGR,
            1,
            cv2.LINE_AA,
        )

    if show_masks:
        cv2.addWeighted(mask_overlay, mask_opacity, display_img, 1.0, 0, display_img)

    focus_label = selected_quadrant if selected_quadrant != "All" else "Full Arch"
    draw_overlay_card(
        display_img,
        16,
        16,
        225,
        "Case Overview",
        [
            f"Case  {case_id}",
            f"Focus  {focus_label}",
            f"Inference  {infer_time:.2f}s",
        ],
        accent=(31, 162, 225),
    )

    draw_overlay_card(
        display_img,
        img_w - 300,
        16,
        285,
        "Tooth Map",
        build_map_lines(teeth_data, conf_thresh, view_mode),
        accent=(42, 157, 143),
    )

    draw_overlay_card(
        display_img,
        16,
        img_h - 94,
        220,
        "Clinical Summary",
        [
            f"Detected  {len(teeth_data)}",
            f"Needs review  {low_conf_count}",
        ],
        accent=(238, 165, 51),
    )

    return display_img, low_conf_count


with st.container():
    head_left, head_mid, head_btn1, head_btn2 = st.columns([5.3, 1.9, 1.6, 1.7])
    with head_left:
        st.markdown(
            """
            <div class="app-hero">
                <div class="app-title">Dental-AI Workstation</div>
                <div class="app-subtitle">Workstation made by Bnyat Kamaran </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_mid:
        st.markdown(
            f"""
            <div class="case-card">
                <div class="case-card-inner">
                    <div class="case-label">Case ID</div>
                    <div class="case-value">{st.session_state.case_id}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_btn1:
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        if st.button("Reset Session", use_container_width=True):
            reset_session()
            st.rerun()
    with head_btn2:
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        export_csv = None
        if st.session_state.teeth_data:
            export_df = build_findings_df(st.session_state.teeth_data, 0.60)
            export_csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Findings CSV",
            data=export_csv or b"",
            file_name=f"dental_ai_findings_{st.session_state.case_id.replace('#', '').replace('-', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not bool(export_csv),
        )

focus_col, mode_col = st.columns([6, 4])
with focus_col:
    selected_quadrant = st.radio("Focus Area", ["All", "UR", "UL", "LL", "LR"], horizontal=True)
with mode_col:
    view_mode = st.radio("Interface Mode", ["Presentation Mode", "Review Mode"], horizontal=True)

show_masks = False
show_boxes = False
show_full_labels = False
mask_opacity = 0.0
conf_thresh = 0.60
if view_mode == "Review Mode":
    show_masks = True
    show_full_labels = True
    mask_opacity = 0.22

main_left, main_right = st.columns([7.45, 2.55])

with main_right:
    st.markdown(
        """
        <div class="ui-card">
            <div class="ui-card-title">Case Input</div>
            <div class="ui-card-subtitle">Upload one panoramic radiograph for live review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload Radiograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=st.session_state.uploader_key,
    )

    if st.session_state.uploaded_name:
        st.markdown(
            f"""
            <div class="ui-card">
                <div class="uploaded-row">
                    <div>
                        <div class="uploaded-name">{st.session_state.uploaded_name}</div>
                        <div class="uploaded-meta">{human_file_size(st.session_state.uploaded_size)}. Upload another image to replace it.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if view_mode == "Review Mode":
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-title">Viewer Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-subtitle">Review overlays and confidence thresholds.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        show_masks = st.toggle("Show AI masks", value=True)
        show_boxes = st.toggle("Show bounding boxes", value=False)
        show_full_labels = st.toggle("Show full tooth labels", value=True)
        mask_opacity = st.slider("Mask opacity", 0.10, 0.70, 0.22, 0.01)
        conf_thresh = st.slider("Confidence warning threshold", 0.10, 0.95, 0.60, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="ui-card">
                <div class="ui-card-title">Presentation Mode</div>
                <div class="ui-card-subtitle">Minimal overlays for doctor-facing demos. Use Review Mode for deeper inspection.</div>
                <div class="mode-chip">Clean clinical view</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image_sig = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.image_sig != image_sig:
        file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        raw_img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if raw_img is None:
            st.error("Could not decode the uploaded image.")
            st.stop()

        with st.spinner("Processing radiograph"):
            teeth_data, infer_time = process_image(raw_img)
            pathology_data = detect_pathology(raw_img)

        st.session_state.image_sig = image_sig
        st.session_state.raw_img = raw_img
        st.session_state.teeth_data = teeth_data
        st.session_state.pathology_data = pathology_data
        st.session_state.infer_time = infer_time
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_size = len(file_bytes)

raw_img = st.session_state.raw_img
teeth_data = st.session_state.teeth_data or []
pathology_data = st.session_state.pathology_data or []
infer_time = st.session_state.infer_time

if raw_img is not None:
    total_detected = len(teeth_data)
    review_count = sum(1 for tooth in teeth_data if tooth["id_conf"] < conf_thresh)
    focus_count = total_detected if selected_quadrant == "All" else sum(1 for tooth in teeth_data if tooth["quad"] == selected_quadrant)

    if selected_quadrant == "All":
        visible_pathology = pathology_data
    else:
        visible_pathology = [
            det
            for det in pathology_data
            if get_box_quadrant(det["bbox"], raw_img.shape[1], raw_img.shape[0]) == selected_quadrant
        ]

    pathology_summary = summarize_pathology(visible_pathology)
    visible_pathology_count = len(visible_pathology)

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card"><div class="summary-label">Detected teeth</div><div class="summary-value">{total_detected}</div></div>
            <div class="summary-card"><div class="summary-label">Visible pathology</div><div class="summary-value">{visible_pathology_count}</div></div>
            <div class="summary-card"><div class="summary-label">Needs review</div><div class="summary-value">{review_count}</div></div>
            <div class="summary-card"><div class="summary-label">Inference time</div><div class="summary-value">{infer_time:.2f}s</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with main_right:
        st.markdown(
            f"""
            <div class="ui-card">
                <div class="ui-card-title">Case Overview</div>
                <div class="ui-card-subtitle">Current focus: <b>{selected_quadrant}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Visible teeth: <b>{focus_count}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Visible pathology: <b>{visible_pathology_count}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    rendered_img, low_conf_count = render_image(
        raw_img=raw_img,
        teeth_data=teeth_data,
        case_id=st.session_state.case_id,
        infer_time=infer_time,
        selected_quadrant=selected_quadrant,
        view_mode=view_mode,
        show_masks=show_masks,
        show_boxes=show_boxes,
        show_full_labels=show_full_labels,
        mask_opacity=mask_opacity,
        conf_thresh=conf_thresh,
    )
    rendered_img = draw_pathology_boxes(rendered_img, visible_pathology, show_conf=False)
    st.session_state.render_png = encode_png(rendered_img)

    with main_left:
        st.markdown("<div class='viewer-shell'>", unsafe_allow_html=True)
        st.image(rendered_img, channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        findings_df = build_findings_df(teeth_data, conf_thresh)
        st.markdown(
            """
            <div class="findings-shell">
                <div class="ui-card-title">Clinical Findings</div>
                <div class="ui-card-subtitle">Summary first. Open detail only when needed.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_left, summary_mid, summary_right = st.columns(3)
        with summary_left:
            st.metric("Current focus count", focus_count)
        with summary_mid:
            st.metric("Needs review", low_conf_count)
        with summary_right:
            st.metric("Pathology findings", visible_pathology_count)

        st.markdown('<div class="ui-card" style="padding-top:0.8rem;">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-title">Pathology Findings</div>', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-subtitle">Model output from the pathology detector.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if pathology_summary:
            for item in pathology_summary:
                st.write(f"{item['label']}: {item['count']} detected, top confidence {item['top_conf']:.2f}")
        else:
            st.write("No pathology detected in the current focus.")
        st.markdown('</div>', unsafe_allow_html=True)

        show_details_default = view_mode == "Review Mode"
        show_details = st.toggle("Show detailed findings", value=show_details_default, key="show_detailed_findings")

        if show_details:
            st.markdown('<div class="ui-card" style="padding-top:0.8rem;">', unsafe_allow_html=True)
            overview_tab, pathology_tab, ur_tab, ul_tab, ll_tab, lr_tab, review_tab = st.tabs(
                ["Overview", "Pathology", "UR", "UL", "LL", "LR", "Needs Review"]
            )

            with overview_tab:
                overview_df = findings_df.copy()
                if selected_quadrant != "All":
                    overview_df = overview_df[overview_df["Quad"] == selected_quadrant]
                if overview_df.empty:
                    st.info("No findings available for the current focus.")
                else:
                    st.dataframe(overview_df, use_container_width=True, hide_index=True, height=340)

            with pathology_tab:
                pathology_df = build_pathology_df(visible_pathology, raw_img.shape[1], raw_img.shape[0])
                if pathology_df.empty:
                    st.info("No pathology detections for the current focus.")
                else:
                    st.dataframe(pathology_df, use_container_width=True, hide_index=True, height=340)

            for tab, quad in [(ur_tab, "UR"), (ul_tab, "UL"), (ll_tab, "LL"), (lr_tab, "LR")]:
                with tab:
                    quad_df = findings_df[findings_df["Quad"] == quad]
                    if quad_df.empty:
                        st.info(f"No detections for {quad}.")
                    else:
                        st.dataframe(quad_df, use_container_width=True, hide_index=True, height=340)

            with review_tab:
                review_df = findings_df[findings_df["Status"] == "Review"]
                if review_df.empty:
                    st.success("No teeth are below the current confidence threshold.")
                else:
                    st.dataframe(review_df, use_container_width=True, hide_index=True, height=340)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("Turn on detailed findings to review quadrant tables and teeth flagged for review.")

        st.markdown('<div class="ui-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-title">Export Options</div>', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-subtitle">Download the annotated image. Teeth tables stay in the findings panel.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.download_button(
            "Download annotated PNG",
            data=st.session_state.render_png,
            file_name=f"annotated_{st.session_state.case_id.replace('#', '').replace('-', '_')}.png",
            mime="image/png",
            use_container_width=True,
            disabled=st.session_state.render_png is None,
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with main_left:
        empty_img = np.full((700, 1200, 3), 241, dtype=np.uint8)
        cv2.putText(
            empty_img,
            "Awaiting panoramic radiograph upload",
            (235, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.02,
            (103, 113, 126),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            empty_img,
            "The clinical viewer and findings will appear here.",
            (275, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (122, 132, 146),
            2,
            cv2.LINE_AA,
        )
        st.markdown("<div class='viewer-shell'>", unsafe_allow_html=True)
        st.image(empty_img, channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with main_right:
        st.markdown(
            """
            <div class="ui-card">
                <div class="ui-card-title">Doctor Demo Guidance</div>
                <div class="ui-card-subtitle">Presentation Mode shows the cleanest screen. Review Mode reveals masks, boxes, labels, and technical confidence controls.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
