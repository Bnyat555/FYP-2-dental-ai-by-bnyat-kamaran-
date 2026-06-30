from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
DEFAULT_MODEL = PROJECT_ROOT / "models" / "tooth_segmentor_best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "segment_teeth" / "images" / "val"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "segment_teeth" / "labels" / "val"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "seg_localize_val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def polygon_to_mask(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(poly_xy).astype(np.int32)

    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    if len(poly) >= 3:
        cv2.fillPoly(mask, [poly], 1)

    return mask


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None

    return contour.squeeze(1).astype(np.float32)


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    area_b = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def build_object_from_mask(mask: np.ndarray, score: float, origin: str) -> dict | None:
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        return None

    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None

    poly = mask_to_polygon(mask)
    if poly is None:
        return None

    x1, y1, x2, y2 = bbox
    width = float(x2 - x1 + 1.0)
    height = float(y2 - y1 + 1.0)
    bbox_area = float(width * height)

    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)

    return {
        "mask": mask,
        "poly": poly,
        "bbox": bbox,
        "score": float(score),
        "origin": origin,
        "area": float(area),
        "bbox_area": bbox_area,
        "width": width,
        "height": height,
        "cx": cx,
        "cy": cy,
    }


def load_gt_objects(label_path: Path, h: int, w: int) -> list[dict]:
    if not label_path.exists():
        return []

    objects = []
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        cls = int(round(vals[0]))
        if cls != 0:
            continue

        coords = vals[1:]
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h

        mask = polygon_to_mask(pts, h, w)
        obj = build_object_from_mask(mask, score=1.0, origin="gt")
        if obj is not None:
            objects.append(obj)

    return objects


def load_pred_objects(result, h: int, w: int) -> list[dict]:
    objects = []

    if result.masks is None or result.boxes is None:
        return objects

    polys = result.masks.xy
    scores = result.boxes.conf.detach().cpu().numpy().tolist()

    for poly, score in zip(polys, scores):
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim != 2 or len(poly) < 3:
            continue

        mask = polygon_to_mask(poly, h, w)
        obj = build_object_from_mask(mask, score=float(score), origin="raw")
        if obj is not None:
            objects.append(obj)

    return objects


def duplicate_nms(objects: list[dict], mask_iou_thr: float, box_iou_thr: float) -> tuple[list[dict], int]:
    if not objects:
        return [], 0

    kept = []
    removed = 0

    objects = sorted(objects, key=lambda x: x["score"], reverse=True)

    for obj in objects:
        is_dup = False
        for keep in kept:
            miou = mask_iou(obj["mask"], keep["mask"])
            biou = bbox_iou(obj["bbox"], keep["bbox"])

            if miou >= mask_iou_thr or biou >= box_iou_thr:
                is_dup = True
                break

        if is_dup:
            removed += 1
        else:
            kept.append(obj)

    return kept, removed


def split_mask_by_watershed(mask: np.ndarray) -> list[np.ndarray]:
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return [mask]

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return [mask]

    seed_thresh = 0.45 * float(dist.max())
    sure_fg = (dist > seed_thresh).astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 2:
        return [mask]

    unknown = cv2.subtract(mask, sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    color = np.dstack([mask * 255, mask * 255, mask * 255]).astype(np.uint8)
    ws = cv2.watershed(color, markers.astype(np.int32))

    parts = []
    for label in range(2, int(ws.max()) + 1):
        part = (ws == label).astype(np.uint8)
        if int(part.sum()) > 0:
            parts.append(part)

    if not (2 <= len(parts) <= 3):
        return [mask]

    min_required = max(20, int(0.15 * mask.sum()))
    parts = [p for p in parts if int(p.sum()) >= min_required]

    if not (2 <= len(parts) <= 3):
        return [mask]

    return parts


def split_oversize_objects(
    objects: list[dict],
    split_area_factor: float,
    split_width_factor: float,
) -> tuple[list[dict], int]:
    if len(objects) < 3:
        return objects, 0

    median_bbox_area = float(np.median([o["bbox_area"] for o in objects]))
    median_width = float(np.median([o["width"] for o in objects]))

    out = []
    split_count = 0

    for obj in objects:
        oversize = (
            obj["bbox_area"] >= split_area_factor * median_bbox_area
            or obj["width"] >= split_width_factor * median_width
        )

        if not oversize:
            out.append(obj)
            continue

        parts = split_mask_by_watershed(obj["mask"])
        if len(parts) == 1:
            out.append(obj)
            continue

        new_parts = []
        for part in parts:
            child = build_object_from_mask(part, score=obj["score"] * 0.92, origin="split")
            if child is not None and child["area"] >= 0.20 * obj["area"]:
                new_parts.append(child)

        if len(new_parts) >= 2:
            out.extend(new_parts)
            split_count += 1
        else:
            out.append(obj)

    return out, split_count


def remove_absurd_oversize(objects: list[dict], hard_area_factor: float) -> tuple[list[dict], int]:
    if len(objects) < 3:
        return objects, 0

    median_bbox_area = float(np.median([o["bbox_area"] for o in objects]))
    out = []
    removed = 0

    for obj in objects:
        absurd = obj["bbox_area"] >= hard_area_factor * median_bbox_area and obj["score"] < 0.80
        if absurd:
            removed += 1
        else:
            out.append(obj)

    return out, removed


def match_predictions_to_gt(preds: list[dict], gts: list[dict], iou_thr: float) -> tuple[list[tuple[int, int, float]], int, int, int]:
    pairs = []

    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            iou = mask_iou(pred["mask"], gt["mask"])
            if iou >= iou_thr:
                pairs.append((pi, gi, iou))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_preds = set()
    used_gts = set()
    matches = []

    for pi, gi, iou in pairs:
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matches.append((pi, gi, iou))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp

    return matches, tp, fp, fn


def save_masked_crops(image: np.ndarray, preds: list[dict], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, pred in enumerate(sorted(preds, key=lambda x: x["cx"]), start=1):
        mask = pred["mask"]
        x1, y1, x2, y2 = pred["bbox"].astype(int)

        crop = image[y1:y2 + 1, x1:x2 + 1].copy()
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1].copy()

        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)

        save_path = save_dir / f"tooth_{idx:02d}.png"
        cv2.imwrite(str(save_path), rgba)


def render_overlay(image: np.ndarray, preds: list[dict], gts: list[dict], save_path: Path) -> None:
    canvas = image.copy()

    for gt in gts:
        poly = np.round(gt["poly"]).astype(np.int32)
        cv2.polylines(canvas, [poly], True, (255, 0, 0), 1)

    preds_sorted = sorted(preds, key=lambda x: x["cx"])
    for idx, pred in enumerate(preds_sorted, start=1):
        poly = np.round(pred["poly"]).astype(np.int32)
        color = (0, 255, 0) if pred["origin"] == "raw" else (0, 200, 255)

        cv2.polylines(canvas, [poly], True, color, 2)

        x1, y1, _, _ = pred["bbox"].astype(int)
        label = f"{idx}:{pred['score']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-det", type=int, default=64)
    parser.add_argument("--min-area-ratio", type=float, default=0.00005)
    parser.add_argument("--dup-mask-iou", type=float, default=0.35)
    parser.add_argument("--dup-box-iou", type=float, default=0.55)
    parser.add_argument("--split-area-factor", type=float, default=2.20)
    parser.add_argument("--split-width-factor", type=float, default=1.75)
    parser.add_argument("--hard-area-factor", type=float, default=3.00)
    parser.add_argument("--match-iou", type=float, default=0.50)
    args = parser.parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    annotated_dir = output_dir / "annotated"
    crops_dir = output_dir / "crops"
    reports_dir = output_dir / "reports"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images = list_images(source_dir)

    rows = []
    total_gt = 0
    total_raw = 0
    total_final = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_removed_dups = 0
    total_split = 0
    total_removed_absurd = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_objects = load_gt_objects(label_path, h, w)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            retina_masks=True,
            device=args.device,
            verbose=False,
        )[0]

        raw_objects = load_pred_objects(result, h, w)

        min_area_px = max(16, int(args.min_area_ratio * h * w))
        raw_objects = [o for o in raw_objects if o["area"] >= min_area_px]

        deduped, removed_dups_1 = duplicate_nms(raw_objects, args.dup_mask_iou, args.dup_box_iou)
        split_fixed, split_count = split_oversize_objects(
            deduped,
            split_area_factor=args.split_area_factor,
            split_width_factor=args.split_width_factor,
        )
        deduped_again, removed_dups_2 = duplicate_nms(split_fixed, args.dup_mask_iou, args.dup_box_iou)
        final_objects, removed_absurd = remove_absurd_oversize(deduped_again, args.hard_area_factor)

        matches, tp, fp, fn = match_predictions_to_gt(final_objects, gt_objects, args.match_iou)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_match_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        render_overlay(image, final_objects, gt_objects, annotated_dir / f"{image_path.stem}_seg.png")
        save_masked_crops(image, final_objects, crops_dir / image_path.stem)

        row = {
            "image": image_path.name,
            "gt_count": len(gt_objects),
            "raw_pred_count": len(raw_objects),
            "final_pred_count": len(final_objects),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_match_iou": round(mean_match_iou, 4),
            "removed_duplicates": removed_dups_1 + removed_dups_2,
            "split_oversize_masks": split_count,
            "removed_absurd_oversize": removed_absurd,
        }
        rows.append(row)

        total_gt += len(gt_objects)
        total_raw += len(raw_objects)
        total_final += len(final_objects)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_removed_dups += removed_dups_1 + removed_dups_2
        total_split += split_count
        total_removed_absurd += removed_absurd

        print(
            f"{image_path.name} | "
            f"GT={len(gt_objects)} RAW={len(raw_objects)} FINAL={len(final_objects)} "
            f"TP={tp} FP={fp} FN={fn} "
            f"DUPS_REMOVED={removed_dups_1 + removed_dups_2} "
            f"SPLIT={split_count} "
            f"ABSURD_REMOVED={removed_absurd}"
        )

    csv_path = reports_dir / "localization_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "gt_count",
                "raw_pred_count",
                "final_pred_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_match_iou",
                "removed_duplicates",
                "split_oversize_masks",
                "removed_absurd_oversize",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    overall = {
        "images": len(rows),
        "total_gt_count": total_gt,
        "total_raw_pred_count": total_raw,
        "total_final_pred_count": total_final,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "total_removed_duplicates": total_removed_dups,
        "total_split_oversize_masks": total_split,
        "total_removed_absurd_oversize": total_removed_absurd,
        "summary_csv": str(csv_path),
        "annotated_dir": str(annotated_dir),
        "crops_dir": str(crops_dir),
    }

    json_path = reports_dir / "localization_overall.json"
    json_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print("\nOverall summary")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()