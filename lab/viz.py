"""Vẽ box (đúng = xanh lá, dự đoán = đỏ) và chọn ảnh minh hoạ đúng/sai."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.labels import box_to_pixel, objects_to_boxes  # noqa: E402
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output  # noqa: E402

COLORS = {"gt": (0, 220, 0), "pred": (255, 40, 40)}


def draw(entry, pred=None, coord="norm1000", title=None):
    im = Image.open(ROOT / "data" / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = entry["width"], entry["height"]
    for b in objects_to_boxes(entry["objects"]):
        x1, y1, x2, y2 = box_to_pixel(b["bbox_2d"], "norm1000", W, H)
        d.rectangle([x1, y1, x2, y2], outline=COLORS["gt"], width=2)
        d.text((x1 + 2, max(0, y1 - 11)), "GT " + b["label"], fill=COLORS["gt"])
    for p in pred or []:
        x1, y1, x2, y2 = box_to_pixel(p["bbox"], coord, W, H)
        d.rectangle([x1, y1, x2, y2], outline=COLORS["pred"], width=2)
        d.text((x1 + 2, min(H - 11, y2 + 1)), p["label"], fill=COLORS["pred"])
    if title:
        d.rectangle([0, 0, W, 14], fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
    return im


def per_image_stats(entries, preds, coord="norm1000"):
    rows = []
    for e in entries:
        parsed = parse_output(preds.get(e["image_id"]))
        pr = _to_norm1000(parsed or [], coord, e["width"], e["height"])
        per, _ = _match(_gt_boxes(e), pr)
        tp = sum(v[0] for v in per.values())
        fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        rows.append({"e": e, "pred": parsed, "tp": tp, "fp": fp, "fn": fn, "valid": parsed is not None})
    return rows


def make_gallery(entries, pred_path, out_dir, name, coord="norm1000", n_good=3, n_bad=3):
    preds = {}
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["image_id"]] = r["raw"]
    rows = per_image_stats(entries, preds, coord)
    good = sorted([r for r in rows if r["tp"] > 0 and r["fp"] + r["fn"] == 0],
                  key=lambda r: -r["tp"])[:n_good]
    if len(good) < n_good:
        good += sorted([r for r in rows if r not in good and r["tp"] > 0],
                       key=lambda r: (r["fp"] + r["fn"]) - r["tp"])[: n_good - len(good)]
    bad = sorted(rows, key=lambda r: -(r["fp"] + r["fn"]))[:n_bad]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    tiles = []
    for kind, lst in (("dung", good), ("sai", bad)):
        for r in lst:
            e = r["e"]
            t = f"{name} {kind}: {e['image_id']} TP={r['tp']} FP={r['fp']} FN={r['fn']}"
            im = draw(e, r["pred"], coord, title=t)
            fn = f"{name.lower()}_{kind}_{e['image_id']}.jpg"
            im.save(out_dir / fn, quality=88)
            tiles.append(im)
            captions.append({"file": fn, "kind": kind, "image_id": e["image_id"], "tp": r["tp"],
                             "fp": r["fp"], "fn": r["fn"],
                             "gt": [o["label"] for o in e["objects"]],
                             "pred": [p["label"] for p in (r["pred"] or [])],
                             "raw": preds.get(e["image_id"], "")[:300]})
    if tiles:
        w, h = tiles[0].size
        cols = 3
        rows_n = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (w * cols, h * rows_n), (0, 0, 0))
        for k, im in enumerate(tiles):
            grid.paste(im, ((k % cols) * w, (k // cols) * h))
        grid.save(out_dir / f"{name.lower()}_gallery.jpg", quality=85)
    (out_dir / f"{name.lower()}_captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    return captions


def summary_table(ev_dir):
    rows = []
    for p in sorted(Path(ev_dir).glob("e*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
        rows.append(f"| {d['name']} | {meta.get('trainable_params', 0):,} | "
                    f"{meta.get('peak_vram_gb', '-')} | {meta.get('train_minutes', '-')} | "
                    f"{m['f1']:.3f} | {d['f1_ci95'][0]:.3f}–{d['f1_ci95'][1]:.3f} | "
                    f"{d['fp_per_image']:.2f} | {pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | "
                    f"{pc['manhole']['f1']:.3f} | {d['parse_valid_rate']:.3f} | {d['mean_iou']:.3f} |")
    return "\n".join(rows)
