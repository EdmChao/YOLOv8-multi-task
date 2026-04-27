"""Visualize detection and mask outputs saved by test_multitask.py

Usage examples:
python inference/tools/visualize_outputs.py --detections ./output/result_0_detections.npz --image ./output/result_0_orig.png --out ./output/result_0_vis.png
python inference/tools/visualize_outputs.py --masks ./output/result_0_masks.npz --image ./output/result_0_orig.png --out ./output/result_0_masks_vis.png
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image_path, detections_npz, out_path, box_color=(255,0,0)):
    data = np.load(detections_npz)
    boxes = data.get('boxes')
    scores = data.get('scores')
    classes = data.get('classes')

    img = Image.open(image_path).convert('RGB') if image_path and Path(image_path).exists() else Image.new('RGB', (1280,720), (255,255,255))
    draw = ImageDraw.Draw(img)

    # simple font fallback
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if boxes is None:
        print('No boxes found in', detections_npz)
    else:
        boxes = np.array(boxes)
        scores = np.array(scores) if scores is not None else [None]*len(boxes)
        classes = np.array(classes) if classes is not None else [None]*len(boxes)
        for i, b in enumerate(boxes):
            try:
                x1,y1,x2,y2 = [float(x) for x in b]
            except Exception:
                continue
            draw.rectangle([x1,y1,x2,y2], outline=box_color, width=2)
            label = ''
            if classes is not None and len(classes) > i:
                label += str(int(classes[i]))
            if scores is not None and len(scores) > i and scores[i] is not None:
                label += (f' {scores[i]:.2f}')
            if label:
                draw.text((x1+3,y1+3), label, fill=box_color, font=font)

    img.save(out_path)
    print('Saved visualization to', out_path)


def visualize_masks(image_path, masks_npz, out_path, alpha=0.5):
    data = np.load(masks_npz)
    # expect 'masks' key
    masks = None
    for k in data.files:
        if k == 'masks':
            masks = data[k]
            break
    if masks is None:
        # fallback: take first array
        masks = data[data.files[0]]

    # masks may be (N,H,W) or (H,W,N) or (H,W)
    masks = np.asarray(masks)
    if masks.ndim == 3:
        # (N,H,W) or (H,W,N)
        if masks.shape[0] <= 4:
            nmasks = masks.shape[0]
        else:
            nmasks = masks.shape[0]
    elif masks.ndim == 2:
        nmasks = 1
        masks = masks[None,...]
    else:
        # try transpose
        masks = masks.transpose(2,0,1)
        nmasks = masks.shape[0]

    # base image
    if image_path and Path(image_path).exists():
        base = Image.open(image_path).convert('RGBA')
    else:
        h,w = masks.shape[1], masks.shape[2]
        base = Image.new('RGBA', (w,h), (255,255,255,255))

    base_arr = np.array(base).astype(np.uint8)

    # overlay masks with distinct colors
    colors = [(255,0,0, int(255*alpha)), (0,255,0, int(255*alpha)), (0,0,255, int(255*alpha)), (255,255,0, int(255*alpha))]
    overlay = base.copy()
    for i in range(min(nmasks, len(colors))):
        m = masks[i]
        # normalize mask to 0..255
        try:
            mm = (m > m.mean()).astype(np.uint8) * 255
        except Exception:
            mm = (np.asarray(m) > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mm).convert('L')
        color = colors[i % len(colors)]
        color_img = Image.new('RGBA', mask_img.size, color)
        overlay.paste(color_img, (0,0), mask_img)

    # composite
    comp = Image.alpha_composite(base.convert('RGBA'), overlay)
    comp.save(out_path)
    print('Saved mask visualization to', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detections', help='path to detections .npz file (boxes,scores,classes)')
    p.add_argument('--masks', help='path to masks .npz file')
    p.add_argument('--image', help='optional original image to overlay')
    p.add_argument('--out', required=True, help='output image path')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if args.detections:
        draw_boxes(args.image, args.detections, outp)
    elif args.masks:
        visualize_masks(args.image, args.masks, outp)
    else:
        print('Provide --detections or --masks')
