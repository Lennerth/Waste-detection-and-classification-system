from __future__ import annotations

import gc
import os
import re
import sys
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import cv2
import numpy as np
from PIL import Image


T = TypeVar("T")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SAM2_CHECKPOINT_URLS: Dict[str, str] = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}

VALID_MODES = {"grid", "gdino"}


# ---------------------------------------------------------------------------
# Prompting helpers
# ---------------------------------------------------------------------------

def prompt_value(
    prompt: str,
    default: T,
    parser: Callable[[str], T],
    validator: Callable[[T], bool] | None = None,
    validator_message: str | None = None,
) -> T:
    """Prompt in terminal and fall back to default on blank/invalid input."""
    raw = input(f"{prompt} [default: {default}]: ")
    if raw.strip() == "":
        return default
    try:
        value = parser(raw.strip())
    except ValueError:
        print(f"Invalid value '{raw}'. Using default: {default}")
        return default
    if validator is not None and not validator(value):
        message = validator_message or "Value out of range."
        print(f"{message} Using default: {default}")
        return default
    return value


def prompt_path(prompt: str, default: str) -> str:
    return prompt_value(prompt, default, str)


def prompt_positive_int(prompt: str, default: int) -> int:
    return prompt_value(
        prompt=prompt,
        default=default,
        parser=int,
        validator=lambda v: v > 0,
        validator_message="Value must be > 0.",
    )


def prompt_non_negative_int(prompt: str, default: int) -> int:
    return prompt_value(
        prompt=prompt,
        default=default,
        parser=int,
        validator=lambda v: v >= 0,
        validator_message="Value must be >= 0.",
    )


def prompt_unit_float(prompt: str, default: float) -> float:
    return prompt_value(
        prompt=prompt,
        default=default,
        parser=float,
        validator=lambda v: 0.0 <= v <= 1.0,
        validator_message="Value must be in range [0.0, 1.0].",
    )


def prompt_mode(prompt: str, default: str) -> str:
    return prompt_value(
        prompt=prompt,
        default=default,
        parser=str,
        validator=lambda v: v in VALID_MODES,
        validator_message=f"Mode must be one of: {', '.join(sorted(VALID_MODES))}.",
    )


# ---------------------------------------------------------------------------
# Resource management helpers
# ---------------------------------------------------------------------------

ESTIMATED_PEAK_MB: Dict[str, int] = {
    "sam2.1_hiera_tiny.pt": 400,
    "sam2.1_hiera_small.pt": 600,
    "sam2.1_hiera_base_plus.pt": 900,
    "sam2.1_hiera_large.pt": 1800,
    "sam2_hiera_tiny.pt": 400,
    "sam2_hiera_small.pt": 600,
    "sam2_hiera_base_plus.pt": 900,
    "sam2_hiera_large.pt": 1800,
}


def configure_cpu_threads(reserved_cpus: int) -> int:
    """Limit PyTorch threads, reserving cores for the OS and other processes."""
    import torch

    total = os.cpu_count() or 1
    usable = max(1, total - reserved_cpus)
    torch.set_num_threads(usable)
    return usable


def check_ram_availability(
    checkpoint_name: str, ram_usage_fraction: float, extra_mb: int = 0
) -> None:
    """Warn if estimated peak memory exceeds the usable share of free RAM."""
    try:
        import psutil
    except ImportError:
        print("  (psutil not installed -- skipping RAM pre-check)")
        return

    mem = psutil.virtual_memory()
    available = mem.available
    usable = available * ram_usage_fraction
    peak_est_mb = ESTIMATED_PEAK_MB.get(Path(checkpoint_name).name, 500) + extra_mb
    peak_est_bytes = peak_est_mb * 1e6

    print(
        f"RAM: {available / 1e9:.1f} GB available, "
        f"using up to {ram_usage_fraction * 100:.0f}% = {usable / 1e9:.1f} GB"
    )

    if usable < peak_est_bytes:
        print(
            f"WARNING: Estimated peak usage (~{peak_est_mb} MB) may exceed "
            f"usable RAM ({usable / 1e6:.0f} MB).\n"
            f"  Consider closing other applications or reducing max_image_size."
        )


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def normalize_sam2_config_name(config_value: str) -> str:
    """Map short config names to Hydra package config paths."""
    value = config_value.strip()
    if "/" in value or "\\" in value:
        return value
    if value.startswith("sam2.1_") and value.endswith(".yaml"):
        return f"configs/sam2.1/{value}"
    if value.startswith("sam2_") and value.endswith(".yaml"):
        return f"configs/sam2/{value}"
    return value


def _download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(f"\r  Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
    else:
        mb_done = block_num * block_size / 1e6
        sys.stdout.write(f"\r  Downloading: {mb_done:.1f} MB")
    sys.stdout.flush()


def ensure_checkpoint(checkpoint_path: str) -> str:
    """If checkpoint file is missing but a known URL exists, download it."""
    path = Path(checkpoint_path)
    if path.is_file():
        return checkpoint_path

    filename = path.name
    url = SAM2_CHECKPOINT_URLS.get(filename)
    if url is None:
        return checkpoint_path

    print(f"Checkpoint '{filename}' not found locally. Downloading...")
    try:
        urllib.request.urlretrieve(url, str(path), reporthook=_download_progress_hook)
        print()
        print(f"  Saved checkpoint to {path}")
    except Exception as exc:
        print(f"\n  Download failed: {exc}")
    return checkpoint_path


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------

def resize_for_inference(
    image: np.ndarray, max_side: int
) -> tuple[np.ndarray, float]:
    """Resize image so longest side <= max_side. Returns (resized, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def upscale_masks(
    masks: List[Dict[str, object]], original_h: int, original_w: int
) -> List[Dict[str, object]]:
    """Resize each mask's segmentation back to original image dimensions."""
    out: List[Dict[str, object]] = []
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            out.append(dict(m))
            continue
        seg_arr = np.asarray(seg, dtype=np.uint8) * 255
        seg_up = cv2.resize(
            seg_arr, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )
        new_m = dict(m)
        new_m["segmentation"] = seg_up > 127
        out.append(new_m)
    return out


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def stable_random_color(rng: np.random.Generator) -> np.ndarray:
    """Generate a bright BGR color for overlays."""
    return rng.integers(low=64, high=256, size=3, dtype=np.uint8)


def overlay_masks(
    image_bgr: np.ndarray,
    masks: Sequence[Dict[str, object]],
    alpha: float,
    random_seed: int,
) -> np.ndarray:
    """Overlay SAM masks on image with deterministic random colors."""
    if not masks:
        return image_bgr.copy()

    result = image_bgr.copy()
    sorted_masks = sorted(
        masks, key=lambda x: float(x.get("area", 0.0)), reverse=True
    )
    rng = np.random.default_rng(random_seed)

    for mask_info in sorted_masks:
        segmentation = mask_info.get("segmentation")
        if segmentation is None:
            continue
        mask = np.asarray(segmentation, dtype=bool)
        if mask.ndim != 2:
            continue
        if mask.shape[0] != result.shape[0] or mask.shape[1] != result.shape[1]:
            continue

        color = stable_random_color(rng)
        region = result[mask]
        blended = ((1.0 - alpha) * region + alpha * color).astype(np.uint8)
        result[mask] = blended

    return result


# ---------------------------------------------------------------------------
# Mask-level NMS (suppress heavily overlapping masks)
# ---------------------------------------------------------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of the same shape."""
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def suppress_overlapping_masks(
    masks: List[Dict[str, object]],
    iou_threshold: float,
) -> List[Dict[str, object]]:
    """Remove near-duplicate masks.  When two masks overlap above *iou_threshold*,
    the larger one is discarded (the smaller, tighter mask is kept)."""
    if iou_threshold >= 1.0 or len(masks) <= 1:
        return masks

    sorted_masks = sorted(
        masks, key=lambda m: int(m.get("area", 0)), reverse=False
    )

    keep: List[Dict[str, object]] = []
    for candidate in sorted_masks:
        seg_c = np.asarray(candidate.get("segmentation"), dtype=bool)
        is_duplicate = False
        for kept in keep:
            seg_k = np.asarray(kept.get("segmentation"), dtype=bool)
            if seg_c.shape == seg_k.shape and mask_iou(seg_c, seg_k) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(candidate)
    return keep


# ---------------------------------------------------------------------------
# Model building -- grid mode (SAM2AutomaticMaskGenerator)
# ---------------------------------------------------------------------------

def build_generator(
    model_cfg: str,
    checkpoint_path: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
):
    """Build a CPU SAM2 automatic mask generator."""
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    device = torch.device("cpu")
    model = build_sam2(model_cfg, checkpoint_path, device=device)
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )


# ---------------------------------------------------------------------------
# Model building -- gdino mode (Grounding DINO + SAM2ImagePredictor)
# ---------------------------------------------------------------------------

def build_grounding_dino(
    model_name: str,
) -> Tuple[Any, Any]:
    """Load Grounding DINO model and processor on CPU (natively supported)."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name, dtype=torch.float32
    )
    model.eval()
    return model, processor


def detect_objects_gdino(
    model: Any,
    processor: Any,
    image_pil: Image.Image,
    text_prompt: str,
    threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> List[List[float]]:
    """Run Grounding DINO and return bounding boxes [[x1,y1,x2,y2], ...] in pixel coords."""
    import torch

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    boxes = results["boxes"].tolist()
    return boxes


def build_predictor(model_cfg: str, checkpoint_path: str) -> Any:
    """Build a CPU SAM2ImagePredictor."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cpu")
    sam_model = build_sam2(model_cfg, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam_model)


def predict_masks_from_boxes(
    predictor: Any,
    image_rgb: np.ndarray,
    boxes: List[List[float]],
) -> List[Dict[str, object]]:
    """Use SAM2ImagePredictor to segment each bounding box into a mask."""
    import torch

    predictor.set_image(image_rgb)
    masks_out: List[Dict[str, object]] = []

    for box in boxes:
        box_np = np.array(box, dtype=np.float32)
        with torch.no_grad():
            mask_preds, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=False,
            )
        if mask_preds is not None and len(mask_preds) > 0:
            best_mask = mask_preds[0]
            best_score = float(scores[0]) if scores is not None else 0.0
            masks_out.append({
                "segmentation": best_mask.astype(bool),
                "area": int(best_mask.sum()),
                "predicted_iou": best_score,
                "bbox": box,
            })

    predictor.reset_predictor()
    return masks_out


# ---------------------------------------------------------------------------
# Run folder management
# ---------------------------------------------------------------------------

def next_run_dir(output_dir: Path) -> Path:
    """Return output/run_N where N is the next available integer."""
    existing = [
        int(m.group(1))
        for d in output_dir.iterdir()
        if d.is_dir()
        for m in [re.match(r"^run_(\d+)$", d.name)]
        if m
    ]
    next_num = max(existing, default=0) + 1
    return output_dir / f"run_{next_num}"


def write_params_file(
    run_dir: Path, params: Dict[str, Any], image_count: int
) -> Path:
    """Write a human-readable parameters.txt into the run folder."""
    path = run_dir / "parameters.txt"
    lines = [
        f"SAM2 Run Parameters",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Images in batch: {image_count}",
        "",
    ]
    for key, value in params.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def collect_image_paths(input_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> int:
    import torch

    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print("SAM2 CPU batch mask overlay")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print("Leave prompt blank (or type only spaces) to use defaults.\n")

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print("No images found in input folder. Add images and run again.")
        return 0

    print(f"Found {len(image_paths)} image(s) in input folder.\n")

    # --- mode selection ---
    mode = prompt_mode("detection_mode (grid / gdino)", "gdino")

    # --- shared SAM2 parameters ---
    model_cfg = prompt_path(
        "SAM2 config path", "configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    checkpoint_path = prompt_path("SAM2 checkpoint path", "sam2.1_hiera_base_plus.pt")
    max_image_size = prompt_positive_int("max_image_size (longest side px)", 1024)
    overlay_alpha = prompt_unit_float("overlay_alpha", 0.35)
    random_seed = prompt_value("overlay_random_seed", 42, int)
    nms_iou_threshold = prompt_value(
        "mask_nms_iou_threshold (0.0-1.0, 1.0=disabled)", 0.80, float,
        validator=lambda v: 0.0 <= v <= 1.0,
        validator_message="Value must be in range [0.0, 1.0].",
    )

    # --- mode-specific parameters ---
    grid_params: Dict[str, Any] = {}
    gdino_params: Dict[str, Any] = {}

    if mode == "grid":
        grid_params["points_per_side"] = prompt_positive_int("points_per_side", 16)
        grid_params["pred_iou_thresh"] = prompt_unit_float("pred_iou_thresh", 0.80)
        grid_params["stability_score_thresh"] = prompt_unit_float(
            "stability_score_thresh", 0.90
        )
        grid_params["crop_n_layers"] = prompt_non_negative_int("crop_n_layers", 0)
        grid_params["crop_n_points_downscale_factor"] = prompt_positive_int(
            "crop_n_points_downscale_factor", 2
        )
        grid_params["min_mask_region_area"] = prompt_non_negative_int(
            "min_mask_region_area", 100
        )
    elif mode == "gdino":
        gdino_params["gdino_model"] = prompt_path(
            "Grounding DINO model", "IDEA-Research/grounding-dino-base"
        )
        gdino_params["detection_prompt"] = prompt_path(
            "detection_prompt", "construction waste."
        )
        gdino_params["threshold"] = prompt_value(
            "threshold (0.0-1.0)", 0.25, float,
            validator=lambda v: 0.0 <= v <= 1.0,
            validator_message="Value must be in range [0.0, 1.0].",
        )
        gdino_params["text_threshold"] = prompt_value(
            "text_threshold (0.0-1.0)", 0.25, float,
            validator=lambda v: 0.0 <= v <= 1.0,
            validator_message="Value must be in range [0.0, 1.0].",
        )

    # --- resource management ---
    total_cpus = os.cpu_count() or 1
    reserved_cpus = prompt_non_negative_int(
        f"reserved_cpu_cores (you have {total_cpus})", 1
    )
    ram_fraction = prompt_value(
        "max_ram_usage_fraction (0.0-1.0)",
        0.90,
        float,
        validator=lambda v: 0.0 < v <= 1.0,
        validator_message="Value must be in range (0.0, 1.0].",
    )

    used_threads = configure_cpu_threads(reserved_cpus)
    print(f"\nCPU threads: using {used_threads} of {total_cpus} "
          f"({reserved_cpus} reserved)")

    checkpoint_path = ensure_checkpoint(checkpoint_path)
    gdino_extra_mb = 500 if mode == "gdino" else 0
    check_ram_availability(checkpoint_path, ram_fraction, extra_mb=gdino_extra_mb)

    # --- run folder and parameters ---
    run_dir = next_run_dir(output_dir)
    run_dir.mkdir(parents=True)
    print(f"Run output   : {run_dir}\n")

    run_params: Dict[str, Any] = {
        "mode": mode,
        "model_cfg": model_cfg,
        "checkpoint_path": checkpoint_path,
        "max_image_size": max_image_size,
        "mask_nms_iou_threshold": nms_iou_threshold,
        "overlay_alpha": overlay_alpha,
        "overlay_random_seed": random_seed,
        "reserved_cpu_cores": reserved_cpus,
        "max_ram_usage_fraction": ram_fraction,
    }
    run_params.update(grid_params)
    run_params.update(gdino_params)
    params_path = write_params_file(run_dir, run_params, len(image_paths))
    print(f"Parameters saved to: {params_path.name}")

    # --- model initialization ---
    mask_generator = None
    sam2_predictor = None
    gd_model = None
    gd_processor = None

    try:
        if mode == "grid":
            print("Loading SAM2 automatic mask generator...")
            mask_generator = build_generator(
                model_cfg=normalize_sam2_config_name(model_cfg),
                checkpoint_path=checkpoint_path,
                points_per_side=grid_params["points_per_side"],
                pred_iou_thresh=grid_params["pred_iou_thresh"],
                stability_score_thresh=grid_params["stability_score_thresh"],
                crop_n_layers=grid_params["crop_n_layers"],
                crop_n_points_downscale_factor=grid_params[
                    "crop_n_points_downscale_factor"
                ],
                min_mask_region_area=grid_params["min_mask_region_area"],
            )
        elif mode == "gdino":
            print("Loading Grounding DINO model...")
            gd_model, gd_processor = build_grounding_dino(
                gdino_params["gdino_model"]
            )
            print("Loading SAM2 image predictor...")
            sam2_predictor = build_predictor(
                model_cfg=normalize_sam2_config_name(model_cfg),
                checkpoint_path=checkpoint_path,
            )
    except Exception as exc:
        print(f"Failed to initialize models: {exc}")
        traceback.print_exc()
        return 1

    # --- image processing loop ---
    success_count = 0
    fail_count = 0
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  Failed to read image: {image_path}")
            fail_count += 1
            continue

        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        small_rgb, scale = resize_for_inference(image_rgb, max_image_size)
        if scale < 1.0:
            print(
                f"  Resized {orig_w}x{orig_h} -> "
                f"{small_rgb.shape[1]}x{small_rgb.shape[0]} for inference"
            )

        try:
            with torch.no_grad():
                if mode == "grid":
                    masks = mask_generator.generate(small_rgb)  # type: ignore[union-attr]
                elif mode == "gdino":
                    small_pil = Image.fromarray(small_rgb)
                    boxes = detect_objects_gdino(
                        gd_model,
                        gd_processor,
                        small_pil,
                        gdino_params["detection_prompt"],
                        threshold=gdino_params["threshold"],
                        text_threshold=gdino_params["text_threshold"],
                    )
                    print(f"  Grounding DINO detected {len(boxes)} object(s)")
                    masks = predict_masks_from_boxes(
                        sam2_predictor, small_rgb, boxes  # type: ignore[arg-type]
                    )
                else:
                    masks = []

            if scale < 1.0 and masks:
                masks = upscale_masks(masks, orig_h, orig_w)

            if masks and nms_iou_threshold < 1.0:
                before = len(masks)
                masks = suppress_overlapping_masks(masks, nms_iou_threshold)
                suppressed = before - len(masks)
                if suppressed > 0:
                    print(f"  Mask NMS: kept {len(masks)}/{before} "
                          f"(suppressed {suppressed} overlapping)")

            overlay = overlay_masks(
                image_bgr=image_bgr,
                masks=masks,
                alpha=overlay_alpha,
                random_seed=random_seed,
            )
        except Exception:
            print(f"  Failed for {image_path.name}:")
            traceback.print_exc()
            fail_count += 1
            del small_rgb, image_rgb, image_bgr
            gc.collect()
            continue

        output_name = f"{image_path.stem}_overlay.png"
        output_path = run_dir / output_name
        saved = cv2.imwrite(str(output_path), overlay)
        if not saved:
            print(f"  Failed to write output: {output_path}")
            fail_count += 1
        else:
            print(f"  Saved: {output_path.name}")
            success_count += 1

        del small_rgb, image_rgb, image_bgr, overlay, masks
        gc.collect()

    print("\nDone.")
    print(f"Total images : {len(image_paths)}")
    print(f"Succeeded    : {success_count}")
    print(f"Failed       : {fail_count}")
    print(f"Output folder: {run_dir}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
