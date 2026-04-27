"""
Multi-task YOLO inference test script.

Purpose:
- Run inference with a YOLO model that performs driving area segmentation,
  lane segmentation, and object detection.

Placeholders:
- MODEL_PATH: path to your .pt model file (replace <PLACEHOLDER> below)
- SOURCE: path to images or directory to run inference on
- OUT_DIR: directory where annotated images and raw outputs are saved

Assumptions:
- You will run this script from the correct project environment and CWD.
- Do NOT rely on sys.path.insert here; run from your repo root or set PYTHONPATH.

Examples:
python inference/test_multitask.py \
  --model <PATH_TO_MODEL.pt> \
  --source <PATH_TO_IMAGES_OR_DIR> \
  --out-dir <OUTPUT_DIR>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics import YOLO


def save_annotated(res, out_path):
	"""Save the YOLO plotted/annotated image for a single result.
	Uses `res.plot()` which is provided by ultralytics Results object.
	"""
	try:
		img = res.plot()  # typically ndarray (RGB)
		Image.fromarray(img).save(out_path)
	except Exception as e:
		print(f"Warning: failed to save annotated image to {out_path}: {e}")


def extract_raw(res):
	"""Extract raw boxes/masks from a result. Returns a tuple (meta_dict, masks_np_or_None).
	meta_dict is JSON-serializable except masks (we save masks separately as .npz).
	"""
	meta = {}
	masks_np = None

	# Boxes (xyxy), confidences, classes
	try:
		boxes = getattr(res.boxes, "xyxy", None)
		confs = getattr(res.boxes, "conf", None)
		clss = getattr(res.boxes, "cls", None)

		def to_list(x):
			if x is None:
				return []
			try:
				return x.cpu().numpy().tolist()
			except Exception:
				try:
					return np.asarray(x).tolist()
				except Exception:
					return []

		meta["boxes"] = to_list(boxes)
		meta["scores"] = to_list(confs)
		meta["classes"] = to_list(clss)
	except Exception as e:
		meta["boxes"] = []
		meta["scores"] = []
		meta["classes"] = []

	# Masks (if present)
	try:
		masks = getattr(res, "masks", None)
		if masks is not None:
			masks_data = getattr(masks, "data", None)
			if masks_data is not None:
				try:
					masks_np = masks_data.cpu().numpy()
				except Exception:
					masks_np = np.asarray(masks_data)
				meta["masks_shape"] = masks_np.shape
			else:
				meta["masks_shape"] = None
		else:
			meta["masks_shape"] = None
	except Exception:
		meta["masks_shape"] = None

	# Optionally include image size / source info
	try:
		wh = getattr(res, "orig_shape", None)
		if wh is not None:
			meta["orig_shape"] = wh
	except Exception:
		pass

	return meta, masks_np


def run(args):

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	print("[INFO] Input args:")
	for k, v in vars(args).items():
		print(f"  {k}: {v}")

	print(f"Loading model: {args.model}")
	model = YOLO(args.model)

	print(f"Running inference on source: {args.source}")
	results = model.predict(
		source=args.source,
		imgsz=args.imgsz,
		device=args.device,
		conf=args.conf,
		iou=args.iou,
		save=False,
		verbose=True,
	)

	for idx, res in enumerate(results):
		# Debug: print result summary so we can inspect types and contents
		try:
			res_type = type(res)
			print(f"[DBG] idx={idx} res_type={res_type}")
			# If it's a list, show length and sample repr
			if isinstance(res, list):
				print(f"[DBG] results[{idx}] is list length={len(res)} repr_sample={str(res)[:200]}")
			else:
				# Try to print boxes/masks brief info
				boxes = getattr(res, 'boxes', None)
				masks = getattr(res, 'masks', None)
				try:
					if boxes is not None:
						bx = getattr(boxes, 'xyxy', None)
						print(f"[DBG] boxes present, xyxy type={type(bx)}")
				except Exception:
					print("[DBG] error reading boxes")
				try:
					if masks is not None:
						md = getattr(masks, 'data', None)
						print(f"[DBG] masks present, data_type={type(md)}")
						try:
							shape = md.cpu().numpy().shape if hasattr(md, 'cpu') else (np.asarray(md).shape)
							print(f"[DBG] masks.data shape={shape}")
						except Exception:
							pass
				except Exception:
					print("[DBG] error reading masks")
		except Exception as e:
			print(f"[DBG] failed to summarize result {idx}: {e}")
		base_name = f"result_{idx}"
		annotated_path = out_dir / f"{base_name}_annotated.png"

		# Handle case where res is a list (should be a Results object)
		if isinstance(res, list):
			print(f"[WARN] results[{idx}] is a list, not a Results object. Skipping annotated image save.")
		else:
			try:
				save_annotated(res, annotated_path)
				print(f"[INFO] Saved annotated image: {annotated_path}")
			except Exception as e:
				print(f"[ERROR] Failed to save annotated image to {annotated_path}: {e}")

		# Save original image if available
		try:
			orig = getattr(res, "orig_img", None)
			if orig is not None:
				Image.fromarray(orig).save(out_dir / f"{base_name}_orig.png")
		except Exception:
			pass

		# Always extract meta, even if res is a list (will be empty)
		try:
			meta, masks_np = extract_raw(res)
		except Exception as e:
			print(f"[ERROR] extract_raw failed: {e}")
			meta, masks_np = {"error": str(e)}, None

		# Save masks separately if present
		if masks_np is not None:
			masks_file = out_dir / f"{base_name}_masks.npz"
			np.savez_compressed(masks_file, masks=masks_np)
			meta["masks_file"] = str(masks_file)
		else:
			meta["masks_file"] = None

		# Save JSON of meta (boxes, classes, shapes)
		json_file = out_dir / f"{base_name}_raw.json"
		with open(json_file, "w") as f:
			json.dump(meta, f, indent=2)

		# Also save a compact numpy file containing boxes & scores & classes if available
		try:
			boxes = meta.get("boxes", [])
			scores = meta.get("scores", [])
			classes = meta.get("classes", [])
			npz_file = out_dir / f"{base_name}_detections.npz"
			np.savez_compressed(npz_file, boxes=np.array(boxes), scores=np.array(scores), classes=np.array(classes))
		except Exception:
			pass

		print(f"[INFO] Saved outputs for {base_name} -> {out_dir}")


def parse_args():
	p = argparse.ArgumentParser(description="Test multitask YOLO inference.")

	p.add_argument("--model", required=True, help="<PLACEHOLDER: path to model .pt>")
	p.add_argument("--source", required=True, help="<PLACEHOLDER: input image or directory>")
	p.add_argument("--out-dir", required=True, help="<PLACEHOLDER: output directory>")
	p.add_argument("--imgsz", type=tuple, default=(384,672), help="image size (h, w) as a tuple to pass to model")
	p.add_argument("--device", default=0, help="device id or string (e.g. 0 or 'cpu')")
	p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
	p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")

	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(args)

#python test_multitask.py --model ./models/v4.pt --source ./test_imgs/638ec620117f75ec4.jpg --out-dir ./output 