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

	# Extract boxes, scores, and classes using the .boxes attribute
	try:
		boxes_obj = res.boxes
		if boxes_obj is not None:
			meta['boxes'] = boxes_obj.xyxy.cpu().numpy().tolist() if boxes_obj.xyxy is not None else []
			meta['scores'] = boxes_obj.conf.cpu().numpy().tolist() if boxes_obj.conf is not None else []
			meta['classes'] = boxes_obj.cls.cpu().numpy().tolist() if boxes_obj.cls is not None else []
		else:
			meta['boxes'] = []
			meta['scores'] = []
			meta['classes'] = []
	except Exception as e:
		print(f"[WARN] Failed to extract boxes: {e}")
		meta['boxes'] = []
		meta['scores'] = []
		meta['classes'] = []

	# Extract masks using the .masks attribute
	try:
		masks = res.masks
		if masks is not None and masks.data is not None:
			masks_np = masks.data.cpu().numpy()
			meta['masks_shape'] = masks_np.shape
		else:
			meta['masks_shape'] = None
	except Exception as e:
		print(f"[WARN] Failed to extract masks: {e}")
		meta['masks_shape'] = None

	# Optionally include image size / source info
	try:
		wh = getattr(res, "orig_shape", None)
		if wh is not None:
			meta["orig_shape"] = wh
	except Exception as e:
		print(f"[WARN] Failed to extract orig_shape: {e}")

	return meta, masks_np


def run(args):

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	print("[INFO] Input args:")
	for k, v in vars(args).items():
		print(f"  {k}: {v}")

	print(f"Loading model: {args.model}")
	# Previous load (kept as comment for easy revert):
	# model = YOLO(args.model)
	# Minimal change: if a YAML is provided, build model from YAML then load weights
	try:
		if getattr(args, 'yaml', None):
			print(f"[INFO] Building model from YAML: {args.yaml} and loading weights: {args.model}")
			model = YOLO(args.yaml).load(args.model)
		else:
			model = YOLO(args.model)
	except Exception as e:
		print(f"[WARN] failed to load model via YAML+weights approach: {e}\nFalling back to YOLO(args.model)")
		model = YOLO(args.model)
	# Debug: print class name mapping from the loaded model
	try:
		print('[DBG] model.names:', getattr(model, 'names', None))
	except Exception as e:
		print(f"[WARN] couldn't read model.names: {e}")

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

	# Print full results object before any warnings/info
	try:
		print('[DBG] Full results object:')
		print(repr(results))
	except Exception as e:
		print(f"[DBG] failed to print full results: {e}")

	# Diagnostic: access first result directly
	try:
		if len(results) == 0:
			print('[WARN] results is empty')
			return
		r0 = results[0]
		if isinstance(r0, list) and len(r0) >= 1:
			print('[DBG] Unwrapping first result list')
			r0 = r0[0]

		print(f"[DBG] first result type: {type(r0)}")
		# Access boxes and masks directly
		try:
			boxes = r0.boxes
			print(f"[DBG] boxes attribute type: {type(boxes)}")
		except Exception as e:
			print(f"[WARN] could not access r0.boxes: {e}")
			boxes = None
		try:
			masks = r0.masks
			print(f"[DBG] masks attribute type: {type(masks)}")
		except Exception as e:
			print(f"[WARN] could not access r0.masks: {e}")
			masks = None

		# Try plotting annotated image
		try:
			annotated_img = r0.plot()
			if annotated_img is None:
				print('[WARN] r0.plot() returned None')
			else:
				out_file = out_dir / 'result_0_annotated.png'
				try:
					if hasattr(annotated_img, 'save'):
						annotated_img.save(out_file)
					else:
						Image.fromarray(annotated_img).save(out_file)
					print('[INFO] Saved annotated image to', out_file)
				except Exception as e:
					print('[ERROR] failed to save annotated image:', e)
		except Exception as e:
			print('[ERROR] r0.plot() failed:', e)

		# Extract raw and save outputs
		meta, masks_np = extract_raw(r0)
		# Debug: show detected class indices and mapped names (if available)
		try:
			classes_list = meta.get('classes', [])
			if classes_list is None:
				classes_list = []
			mapped_names = []
			if hasattr(model, 'names') and model.names is not None:
				for c in classes_list:
					try:
						ci = int(c)
						mapped_names.append(model.names.get(ci, str(ci)))
					except Exception:
						mapped_names.append(str(c))
			print('[DBG] detected class indices:', classes_list)
			print('[DBG] detected class names :', mapped_names)
		except Exception as e:
			print(f"[WARN] failed to print class names: {e}")
		# Debug: masks presence
		try:
			print('[DBG] masks present:', masks_np is not None)
		except Exception:
			pass
		json_file = out_dir / 'result_0_raw.json'
		with open(json_file, 'w') as f:
			json.dump(meta, f, indent=2)
		print('[INFO] Saved raw JSON to', json_file)

		if masks_np is not None:
			masks_file = out_dir / 'result_0_masks.npz'
			np.savez_compressed(masks_file, masks=masks_np)
			print('[INFO] Saved masks to', masks_file)

		try:
			boxes_list = meta.get('boxes', [])
			scores_list = meta.get('scores', [])
			classes_list = meta.get('classes', [])
			det_file = out_dir / 'result_0_detections.npz'
			np.savez_compressed(det_file, boxes=np.array(boxes_list), scores=np.array(scores_list), classes=np.array(classes_list))
			print('[INFO] Saved detections to', det_file)
		except Exception as e:
			print('[WARN] failed to save detections npz:', e)

	except Exception as e:
		print('[ERROR] diagnostic extraction failed:', e)


def parse_args():
	p = argparse.ArgumentParser(description="Test multitask YOLO inference.")

	p.add_argument("--model", required=True, help="<PLACEHOLDER: path to model .pt>")
	p.add_argument("--source", required=True, help="<PLACEHOLDER: input image or directory>")
	p.add_argument("--out-dir", required=True, help="<PLACEHOLDER: output directory>")
	# Optional: build model from YAML then load weights (.pt)
	p.add_argument("--yaml", required=False, default=None, help="optional: path to model YAML to build model before loading weights")
	p.add_argument("--imgsz", type=tuple, default=(384,672), help="image size (h, w) as a tuple to pass to model")
	p.add_argument("--device", default=0, help="device id or string (e.g. 0 or 'cpu')")
	p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
	p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")

	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(args)

#python test_multitask.py --model ./models/v4.pt --source ./test_imgs/638ec620117f75ec4.jpg --out-dir ./output
#python test_multitask.py --model ./models/v4.pt --source ./test_imgs/e157eb545395c043c.jpg --out-dir ./output
#python test_multitask.py --model ./models/v4s.pt --source ./test_imgs/0027eed2-a6630000.jpg --out-dir ./output
#python test_multitask.py --model ./models/v4s.pt --source ./test_imgs/0027eed2-a6630000.jpg --out-dir ./output --yaml /data2/guest_rui/ztrs_workspace/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml