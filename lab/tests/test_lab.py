"""Test mã của tôi (không cần GPU): nhãn, prompt, reward. Chạy: python -m pytest -q lab/tests"""
import json

import pytest

from lab.labels import box_to_pixel, objects_to_answer, yolo_to_box
from lab.reward import reward
from scripts.score import evaluate, parse_output


def test_yolo_to_box_norm1000():
    b = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.1])
    assert b == {"label": "pothole", "bbox_2d": [400, 450, 600, 550]}


def test_yolo_to_box_clamps_and_pixel():
    assert yolo_to_box("crack", [0.05, 0.5, 0.2, 0.2])["bbox_2d"][0] == 0
    assert yolo_to_box("crack", [0.5, 0.5, 0.5, 0.5], "pixel", 640, 360)["bbox_2d"] == [160, 90, 480, 270]


def test_answer_sorted_and_compact():
    objs = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    a = objects_to_answer(objs)
    assert a.startswith('[{"label":"manhole"') and " " not in a
    assert objects_to_answer([]) == "[]"


def test_roundtrip_scores_perfect():
    e = {"image_id": "x", "width": 640, "height": 360,
         "objects": [{"label": "pothole", "yolo": [0.3, 0.6, 0.05, 0.04]},
                     {"label": "crack", "yolo": [0.7, 0.7, 0.3, 0.02]}]}
    r = evaluate([e], {"x": objects_to_answer(e["objects"])}, n_boot=10)
    assert r["micro"]["f1"] == 1.0
    # nhãn pixel chấm bằng --coord pixel cũng phải hoàn hảo; chấm sai thang thì sụp
    px = objects_to_answer(e["objects"], "pixel", 640, 360)
    assert evaluate([e], {"x": px}, coord="pixel", n_boot=10)["micro"]["f1"] == 1.0
    assert evaluate([e], {"x": px}, coord="norm1000", n_boot=10)["micro"]["f1"] == 0.0


def test_box_to_pixel():
    assert box_to_pixel([500, 500, 1000, 1000], "norm1000", 640, 360) == (320, 180, 640, 360)


GT = [{"label": "pothole", "bbox_2d": [100, 100, 200, 200]},
      {"label": "crack", "bbox_2d": [300, 300, 600, 330]}]


def test_reward_perfect():
    assert reward(json.dumps(GT), GT) == pytest.approx(1.0)


def test_reward_broken_json():
    assert reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert reward("I see a pothole", GT) == 0.0


def test_reward_wrong_class():
    wrong = [{"label": "manhole", "bbox_2d": b["bbox_2d"]} for b in GT]
    assert reward(json.dumps(wrong), GT) == pytest.approx(0.2)


def test_reward_extra_boxes_penalised():
    extra = GT + [{"label": "crack", "bbox_2d": [700, 700, 800, 800]},
                  {"label": "crack", "bbox_2d": [10, 10, 20, 20]}]
    r = reward(json.dumps(extra), GT)
    assert r == pytest.approx(0.2 + 0.8 * 2 / 4)


def test_reward_empty_cases():
    assert reward("[]", []) == 1.0
    assert reward("[]", GT) == pytest.approx(0.2)  # trả rỗng không được thưởng thêm
    assert reward(json.dumps(GT), []) == pytest.approx(0.2)


def test_reward_giant_box_not_hacked():
    giant = [{"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]},
             {"label": "crack", "bbox_2d": [0, 0, 1000, 1000]}]
    assert reward(json.dumps(giant), GT) == pytest.approx(0.2)


def test_reward_fenced_json_ok():
    assert reward("```json\n" + json.dumps(GT) + "\n```", GT) == pytest.approx(1.0)
    assert parse_output("[]") == []
