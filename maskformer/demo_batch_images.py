"""
Batch inference and visualization for all images in a folder.

Example:
python Mask2Former-main/tools/demo_batch_images.py \
  --config Mask2Former-main/configs/flow/flow_swin_tiny_single.yaml \
  --weights output/flow_swin_tiny_merge/model_0005219_7.469.pth \
  --input-dir /home/xdu/演示资料/flowvqa-main/out_dir \
  --output-dir output/vis/batch_num \
  --confidence 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.colormap import colormap

# Make repo importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_single_image_once import add_instance_labels, draw_ordered_bboxes, setup_cfg  # noqa: E402


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return [p for p in input_dir.glob(pattern) if p.suffix.lower() in VALID_EXTS and p.is_file()]


def process_image(
    image_path: Path,
    predictor: DefaultPredictor,
    metadata,
    output_dir: Path,
    bbox_output_dir: Path,
    confidence: float,
    show_score: bool,
    label_style: str,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[Skip] Failed to read image: {image_path}")
        return False

    outputs = predictor(image)
    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    instances = outputs["instances"].to("cpu")
    if instances.has("scores"):
        keep = instances.scores >= confidence
        instances = instances[keep]

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
    vis_bgr = add_instance_labels(vis_bgr, instances, metadata, show_score=show_score)

    bbox_vis = draw_ordered_bboxes(image, instances, color=(0, 0, 255), label_style=label_style)

    dataset_id_to_contiguous = metadata.get("thing_dataset_id_to_contiguous_id", {})
    contiguous_to_dataset = {v: k for k, v in dataset_id_to_contiguous.items()} if dataset_id_to_contiguous else {}
    for i in range(len(instances)):
        cls_id = int(instances.pred_classes[i].item())
        score = float(instances.scores[i].item()) if instances.has("scores") else 0.0
        label = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        ds_id = contiguous_to_dataset.get(cls_id, cls_id)
        print(f"[Detection] {image_path.name}: {label} (dataset_id:{ds_id}) score={score:.4f}")

    out_path = output_dir / f"{image_path.stem}_vis{image_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_bgr)

    bbox_out = bbox_output_dir / f"{image_path.stem}_bbox{image_path.suffix}"
    bbox_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bbox_out), bbox_vis)

    print(f"[Saved] vis={out_path} bbox={bbox_out}")
    return True


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Batch image inference")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pth)")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with images to process")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save visualizations")
    parser.add_argument(
        "--bbox-output-dir",
        type=Path,
        default=None,
        help="Optional directory to save bbox visualizations (defaults to output-dir)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images")
    parser.add_argument("--confidence", type=float, default=0.5, help="Score threshold for visualization")
    parser.add_argument("--show-score", action="store_true", help="Show confidence score with class label")
    parser.add_argument(
        "--label-style",
        default="number",
        choices=["alpha", "number"],
        help="Label style for bbox ordering (alpha: A,B,C...; number: 1,2,3...)",
    )
    parser.add_argument(
        "--test-dataset",
        default="",
        help="Optional dataset name to pull metadata from (defaults to first TEST or TRAIN dataset)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    cfg = setup_cfg(args)
    metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(metadata_name)
    predictor = DefaultPredictor(cfg)

    bbox_output_dir = args.bbox_output_dir or args.output_dir
    images = list_images(args.input_dir, args.recursive)
    if not images:
        print(f"No images with extensions {sorted(VALID_EXTS)} found in {args.input_dir}")
        return

    print(f"Found {len(images)} image(s) in {args.input_dir}. Saving results to {args.output_dir}")
    processed = 0
    for img_path in images:
        processed += int(
            process_image(
                image_path=img_path,
                predictor=predictor,
                metadata=metadata,
                output_dir=args.output_dir,
                bbox_output_dir=bbox_output_dir,
                confidence=args.confidence,
                show_score=args.show_score,
                label_style=args.label_style,
            )
        )

    print(f"Done. Processed {processed} / {len(images)} images.")


if __name__ == "__main__":
    main()
