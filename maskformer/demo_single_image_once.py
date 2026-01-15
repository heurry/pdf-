"""
Single-image inference and visualization.

Example:
python tools/demo_single_image_once.py \
  --config Mask2Former-main/configs/flow/flow_swin_tiny_single.yaml \
  --weights output/flow_swin_tiny_single/model_final.pth \
  --input /path/to/image.png \
  --output output/vis/single_vis.png \
  --confidence 0.5

If your dataset is outside the default datasets/coco path, set FLOWCOCO_ROOT accordingly.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.colormap import colormap

# Make repo importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask2former import add_maskformer2_config


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence
    if args.test_dataset:
        cfg.DATASETS.TEST = (args.test_dataset,)
    cfg.freeze()
    return cfg


def add_instance_labels(image_bgr, instances, metadata, show_score=True):
    """Draw class labels (and optional scores) on the image."""
    if len(instances) == 0:
        return image_bgr
    names = metadata.get("thing_classes", [])
    dataset_id_to_contiguous = metadata.get("thing_dataset_id_to_contiguous_id", {})
    contiguous_to_dataset = {v: k for k, v in dataset_id_to_contiguous.items()} if dataset_id_to_contiguous else {}
    out = image_bgr.copy()
    for i in range(len(instances)):
        cls_id = int(instances.pred_classes[i].item())
        label = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        if show_score and instances.has("scores"):
            score = float(instances.scores[i].item())
            label = f"{label} {score:.2f}"
        # Append dataset id if available to avoid confusion when ids are non-contiguous
        if cls_id in contiguous_to_dataset:
            label = f"{label} (id:{contiguous_to_dataset[cls_id]})"

        (x, y), _ = _get_label_position(instances, i, out.shape)
        print(f"[Detection] {label} (x:{x},y:{y})")

        # Draw text with background for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        bg_tl = (x, y - th - 6)
        bg_br = (x + tw + 4, y + 2)
        cv2.rectangle(out, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.putText(out, label, (x + 2, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _alpha_label_from_index(idx):
    """Return alphabetical labels A, B, C ... Z, AA, AB, ... for a given index."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    n = idx
    while True:
        label = alphabet[n % 26] + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def _get_instance_box(instances, index):
    """Return a usable bounding box (x1, y1, x2, y2) from boxes or masks."""
    if instances.has("pred_boxes") and len(instances.pred_boxes) > index:
        box = instances.pred_boxes.tensor.cpu().numpy()[index]
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w > 1 and h > 1:
            return box.astype(float)
    if instances.has("pred_masks"):
        mask = instances.pred_masks[index].cpu().numpy()
        ys, xs = np.where(mask)
        if len(xs):
            return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
    return None


def _get_label_position(instances, index, image_shape):
    """Compute label anchor position consistent with the class-label logic."""
    box = _get_instance_box(instances, index)
    x = y = None
    if box is not None:
        x = int(box[0])  # left
        y = int(box[1]) - 5  # a bit above the box
    if (x is None or y is None) and instances.has("pred_masks"):
        mask = instances.pred_masks[index].cpu().numpy()
        ys, xs = np.where(mask)
        if len(xs):
            x = int(xs.min())
            y = int(ys.min()) - 5
    if x is None or y is None:
        # spread out overlapping fallbacks to avoid stacking in one corner
        x = 10
        y = 10 + 20 * index

    h, w = image_shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return (x, y), box


def draw_ordered_bboxes(image_bgr, instances, color=(0, 0, 255), label_style="alpha"):
    """
    Draw bounding boxes and mark them with labels ordered from top to bottom.

    label_style: "alpha" for A, B, C...; "number" for 1, 2, 3...
    """
    if len(instances) == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    entries = []
    for idx in range(len(instances)):
        box = _get_instance_box(instances, idx)
        if box is None:
            continue
        entries.append(
            {
                "inst_idx": idx,
                "box": box.astype(int),
                "sort_y": float(box[1]),
            }
        )

    if not entries:
        return image_bgr

    entries = sorted(entries, key=lambda e: (e["sort_y"], e["inst_idx"]))
    out = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    for order_idx, entry in enumerate(entries):
        inst_idx = entry["inst_idx"]
        x1, y1, x2, y2 = entry["box"]
        # cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if label_style == "number":
            label = str(order_idx + 1)
        else:
            label = _alpha_label_from_index(order_idx)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Place label at the top-left inside the box with a small inset.
        text_x = max(0, min(w - tw - 1, x1 + 4))
        text_y = max(th + 2, min(h - 2, y1 + th + 4))

        # Black outline for readability, red text for the requested highlight.
        cv2.putText(out, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        print(f"[BBox Order] {label}: index={inst_idx}, top_y={entry['sort_y']:.1f}")
    return out


def main(args):
    cfg = setup_cfg(args)
    metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(metadata_name)
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.input}")

    outputs = predictor(image)
    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    instances = outputs["instances"].to("cpu")
    # Apply score threshold manually to ensure low-score queries are filtered out
    if instances.has("scores"):
        keep = instances.scores >= args.confidence
        instances = instances[keep]
    # Build consistent colors per class
    names = metadata.get("thing_classes", [])
    base_colors = metadata.get("thing_colors", [])
    fallback_colors = colormap(rgb=True)
    assigned_colors = []
    for i in range(len(instances)):
        cls_id = int(instances.pred_classes[i].item())
        if base_colors and cls_id < len(base_colors):
            color = base_colors[cls_id]
        else:
            color = fallback_colors[cls_id % len(fallback_colors)]
        color = np.array(color, dtype=float)
        if color.max() > 1.0:
            color = color / 255.0
        assigned_colors.append(color)

    vis_output = v.overlay_instances(
        boxes=instances.pred_boxes if instances.has("pred_boxes") else None,
        masks=instances.pred_masks if instances.has("pred_masks") else None,
        labels=None,
        assigned_colors=assigned_colors,
    )
    vis_rgb = vis_output.get_image()
    vis_bgr = vis_rgb[:, :, ::-1]
    vis_bgr = add_instance_labels(vis_bgr, instances, metadata, show_score=args.show_score)

    bbox_vis = draw_ordered_bboxes(image, instances, color=(0, 0, 255), label_style=args.label_style)

    # Print detected classes and scores
    dataset_id_to_contiguous = metadata.get("thing_dataset_id_to_contiguous_id", {})
    contiguous_to_dataset = {v: k for k, v in dataset_id_to_contiguous.items()} if dataset_id_to_contiguous else {}
    for i in range(len(instances)):
        cls_id = int(instances.pred_classes[i].item())
        score = float(instances.scores[i].item()) if instances.has("scores") else 0.0
        label = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        ds_id = contiguous_to_dataset.get(cls_id, cls_id)
        print(f"[Detection] {label} (dataset_id:{ds_id}) score={score:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_bgr)
    print(f"Saved visualization to: {out_path}")

    bbox_out = Path(args.bbox_output) if args.bbox_output else out_path.with_name(out_path.stem + "_bbox" + out_path.suffix)
    bbox_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bbox_out), bbox_vis)
    print(f"Saved bbox ordering visualization to: {bbox_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single image inference")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pth)")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save visualization")
    parser.add_argument("--confidence", type=float, default=0.5, help="Score threshold for visualization")
    parser.add_argument("--show-score", action="store_true", help="Show confidence score with class label")
    parser.add_argument(
        "--label-style",
        default="alpha",
        choices=["alpha", "number"],
        help="Label style for bbox ordering (alpha: A,B,C...; number: 1,2,3...)",
    )
    parser.add_argument(
        "--test-dataset",
        default="",
        help="Optional dataset name to pull metadata from (defaults to first TEST or TRAIN dataset)",
    )
    parser.add_argument(
        "--bbox-output",
        default="",
        help="Optional path to save the bbox-only visualization (defaults to <output> with _bbox suffix)",
    )
    args = parser.parse_args()
    main(args)

# python Mask2Former-main/tools/demo_single_image_once.py \
#   --config Mask2Former-main/configs/flow/flow_swin_tiny_single.yaml \
#   --weights output/flow_swin_tiny_single/model_0003959_7.407.pth \
#   --input /home/xdu/演示资料/pic/21.png \
#   --output output/vis/single_vis.png \
#   --confidence 0.5

# python Mask2Former-main/tools/demo_single_image_once.py \
#   --config Mask2Former-main/configs/flow/flow_swin_tiny_single.yaml \
#   --weights output/flow_swin_tiny_merge/model_0005219_7.469.pth \
#   --input /home/xdu/演示资料/pic/21.png \
#   --output output/vis/multi_vis.png \
#   --confidence 0.9
