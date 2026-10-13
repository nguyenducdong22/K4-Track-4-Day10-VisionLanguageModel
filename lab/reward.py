"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO. Python thuần, test được không cần GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score import iou_xyxy, parse_output  # noqa: E402

IOU_THR = 0.5


def _as_xyxy(b):
    if isinstance(b, dict):
        return b["label"], tuple(float(v) for v in (b.get("bbox") or b.get("bbox_2d")))
    raise TypeError(b)


def reward(completion, gt_boxes):
    """0.2 * JSON hợp lệ + 0.8 * (tổng IoU các cặp khớp) / max(n_gt, n_pred, 1).

    gt_boxes: list[{"label", "bbox_2d" | "bbox"}] trên cùng thang với completion.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    gt = [_as_xyxy(b) for b in gt_boxes]
    pr = [(p["label"], p["bbox"]) for p in pred]
    if not gt and not pr:
        return 1.0
    pairs = sorted(((iou_xyxy(g[1], p[1]), i, j) for i, g in enumerate(gt)
                    for j, p in enumerate(pr) if g[0] == p[0]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pr), 1)
