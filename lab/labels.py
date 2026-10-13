"""Bước 2: nhãn YOLO → JSON (thang 0–1000 hoặc pixel) và prompt. Python thuần, không cần GPU."""
import json

CLASSES = ("pothole", "crack", "manhole")

PROMPT_NORM1000 = (
    "Detect every road damage instance in this dashcam image. "
    "Classes: pothole, crack, manhole. "
    "Answer with ONLY a JSON list and nothing else, in the form "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(0 = left/top edge, 1000 = right/bottom edge). "
    "If there is no damage, answer []."
)

PROMPT_PIXEL = (
    "Detect every road damage instance in this dashcam image. "
    "Classes: pothole, crack, manhole. "
    "Answer with ONLY a JSON list and nothing else, in the form "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integer pixel positions in the {w}x{h} image "
    "(x from 0 to {w}, y from 0 to {h}). "
    "If there is no damage, answer []."
)


def get_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    return PROMPT_PIXEL.replace("{w}", str(width)).replace("{h}", str(height))


def _clamp(v, hi):
    return max(0, min(hi, int(round(v))))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    if coord == "norm1000":
        sx, sy = 1000, 1000
    elif coord == "pixel":
        sx, sy = width, height
    else:
        raise ValueError(coord)
    x1, y1 = _clamp((xc - w / 2) * sx, sx), _clamp((yc - h / 2) * sy, sy)
    x2, y2 = _clamp((xc + w / 2) * sx, sx), _clamp((yc + h / 2) * sy, sy)
    # box cực nhỏ có thể sụp sau khi làm tròn: giữ x2 > x1, y2 > y1 để parse_output không loại
    if x2 <= x1:
        x2 = min(sx, x1 + 1)
        x1 = x2 - 1
    if y2 <= y1:
        y2 = min(sy, y1 + 1)
        y1 = y2 - 1
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixel(bbox, coord="norm1000", width=640, height=360):
    """Quy box về pixel để vẽ."""
    if coord == "pixel":
        return tuple(float(v) for v in bbox)
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)
