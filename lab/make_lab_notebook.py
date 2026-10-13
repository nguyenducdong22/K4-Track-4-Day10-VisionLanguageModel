"""Sinh lab_day10.ipynb đã điền mã (giữ nguyên các mục ## 0. … ## 8. của khung đề)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường\n"
       "Mã chính nằm trong `lab/` (`labels.py`, `vlm.py`, `run.py`, `viz.py`, `reward.py`). "
       "Mỗi lần huấn luyện/suy luận chạy trong **tiến trình riêng** (`!python -m lab.run …`) để VRAM đỉnh đo sạch "
       "và notebook không giữ bộ nhớ GPU."),
    md("## 0. Môi trường (Colab T4)"),
    code(r"""
# Runtime → Change runtime type → T4 GPU
import os, sys, subprocess
from google.colab import drive
drive.mount("/content/drive")
REPO_URL = "https://github.com/nguyenducdong22/K4-Track-4-Day10-VisionLanguageModel"
REPO = "/content/K4-Track-4-Day10-VisionLanguageModel"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
os.chdir(REPO); sys.path.insert(0, REPO)
SRC = "/content/drive/MyDrive/day10/lab_src.txt"   # bản mã lab/ khi chưa push lên GitHub
if not os.path.exists("lab/vlm.py") and os.path.exists(SRC):
    cur = None
    for line in open(SRC, encoding="utf-8").read().splitlines(True):
        if line.startswith("=====FILE: "):
            cur = line[11:].strip().rstrip("=").strip()
            os.makedirs(os.path.dirname(cur), exist_ok=True)
            open(cur, "w", encoding="utf-8").close()
        elif cur:
            open(cur, "a", encoding="utf-8").write(line)
    for d in ("lab", "lab/tests"):
        open(f"{d}/__init__.py", "a").close()
!md5sum lab/*.py lab/tests/*.py
!nvidia-smi -L
!pip -q install -r requirements-colab.txt kagglehub
"""),
    code(r"""
# Lưu work/ (adapter, dự đoán) lên Google Drive để không mất khi session bị ngắt.
USE_DRIVE = True
if USE_DRIVE:
    os.makedirs("/content/drive/MyDrive/day10_work", exist_ok=True)
    if not os.path.islink("work"):
        !rm -rf work
        os.symlink("/content/drive/MyDrive/day10_work", "work")
!ls -la work/
"""),
    code(r"""
# Kaggle: dùng Colab Secrets (KAGGLE_USERNAME, KAGGLE_KEY). KHÔNG viết khoá vào notebook.
import os
try:
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    HAS_KAGGLE = True
except Exception as e:
    print("Không có Colab Secrets Kaggle → tải công khai bằng kagglehub:", type(e).__name__)
    HAS_KAGGLE = False
"""),
    md("## 1. Dữ liệu: tải và chia split"),
    code(r"""
if not os.path.exists("data/splits/meta.json"):
    if HAS_KAGGLE:
        !python scripts/prepare_data.py --kaggle
    else:
        import kagglehub
        src = kagglehub.dataset_download("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")
        !python scripts/prepare_data.py --source "{src}"
!python -m pytest -q
"""),
    code(r"""
# Khám phá dữ liệu: số ảnh/box từng split, kiểm tra không phiên nào nằm ở 2 split, vẽ 6 ảnh có box
import json, collections
from lab.vlm import read_split
from lab.viz import draw
from PIL import Image
meta = json.load(open("data/splits/meta.json"))
print("dataset_version:", meta["dataset_version"])
splits = {s: read_split(s) for s in ("train", "val", "test")}
sess = {s: {e["session"] for e in v} for s, v in splits.items()}
for s, v in splits.items():
    c = collections.Counter(o["label"] for e in v for o in e["objects"])
    print(f"{s:5s} ảnh={len(v):4d} phiên={len(sess[s]):3d} box={dict(c)} ảnh rỗng={sum(not e['objects'] for e in v)}")
assert not (sess["train"] & sess["val"]) and not (sess["train"] & sess["test"]) and not (sess["val"] & sess["test"])
print("OK: không phiên nào xuất hiện ở hai split")
areas = sorted(o["yolo"][2] * o["yolo"][3] for v in splits.values() for e in v for o in e["objects"])
print("diện tích box trung vị (% khung hình):", round(100 * areas[len(areas) // 2], 2))
tiles = [draw(e, title=e["image_id"]) for e in splits["train"][::240][:6]]
grid = Image.new("RGB", (640 * 3, 360 * 2))
for k, im in enumerate(tiles):
    grid.paste(im, ((k % 3) * 640, (k // 3) * 360))
os.makedirs("report/figures", exist_ok=True)
grid.save("report/figures/data_samples.jpg", quality=85)
grid.resize((960, 360))
"""),
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code(r"""
from lab.labels import yolo_to_box, objects_to_answer, box_to_pixel, get_prompt
e = splits["train"][0]
print(e["objects"])
ans = objects_to_answer(e["objects"])
print("answer:", ans)
print("prompt:", get_prompt())
# Kiểm tra quay ngược: 0–1000 → pixel phải trùng box YOLO gốc (sai số làm tròn ≤ 0,64 px)
for o, b in zip(sorted(e["objects"], key=lambda o: (o["yolo"][1] - o["yolo"][3] / 2)), json.loads(ans)):
    xc, yc, w, h = o["yolo"]
    gt_px = ((xc - w / 2) * 640, (yc - h / 2) * 360, (xc + w / 2) * 640, (yc + h / 2) * 360)
    back = box_to_pixel(b["bbox_2d"], "norm1000", 640, 360)
    print(b["label"], [round(v, 1) for v in gt_px], "→", [round(v, 1) for v in back])
!python -m pytest -q lab/tests
"""),
    md("## 3. E0: zero-shot"),
    code(r"""
!python -m lab.run infer --out work/e0.jsonl --split test
"""),
    code("!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json"),
    code(r"""
from scripts.score import parse_output
rows = [json.loads(l) for l in open("work/e0.jsonl")]
for r in rows[:5]:
    print(r["image_id"], "→", repr(r["raw"][:300]), "| parse:", parse_output(r["raw"]) is not None)
"""),
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code(r"""
# encode_example() và make_collate() nằm trong lab/vlm.py.
# Kiểm tra (tiến trình riêng): 220 token ảnh, phần có loss == JSON đáp án + <|im_end|>,
# 252/348 module LoRA, số tham số từng khối và tham số huấn luyện E1/E2/E3.
!python -m lab.inspect_setup
"""),
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code(r"""
# apply_setup(model, targets, train_projector) trong lab/vlm.py:
#   - đóng băng mọi tham số, enable_input_require_grads()
#   - LoRA r=16 α=32 trên regex LLM (252 module) [+ ViT blocks (96) cho E3]
#   - E2/E3: visual.merger + visual.deepstack_merger_list → float32, requires_grad=True
print(json.dumps(json.load(open("report/setup_check.json")), indent=1, ensure_ascii=False)[:2500])
"""),
    md("## 6. Vòng huấn luyện"),
    code(r"""
# Chạy thử nhanh (2 bước trên val, suy luận 8 ảnh) để bắt lỗi trước khi tốn 30 phút
!python -m lab.run train --exp e2 --out work/smoke --split val --max-steps 2
!python -m lab.run infer --ckpt work/smoke --out work/smoke.jsonl --split val --limit 8
!rm -rf work/smoke work/smoke.jsonl
"""),
    code(r"""
# AdamW lr 2e-4, warmup 5% + cosine, fp16 autocast + GradScaler, batch 2 × accum 4, 300 bước,
# gradient checkpointing, lưu adapter (+ projector.pt cho E2/E3) mỗi 100 bước, ghi run.json.
!python -m lab.run train --exp e1 --out work/e1
"""),
    code("!python -m lab.run train --exp e2 --out work/e2"),
    code("!python -m lab.run train --exp e3 --out work/e3"),
    code(r"""
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
for k in ("e1", "e2", "e3", "e4_data_size_0.1"):
    p = f"work/{k}/train_log.json"
    if os.path.exists(p):
        log = json.load(open(p))
        ys = [r["loss"] for r in log]
        sm = [sum(ys[max(0, i - 9):i + 1]) / len(ys[max(0, i - 9):i + 1]) for i in range(len(ys))]
        plt.plot([r["step"] for r in log], sm, label=k)
plt.xlabel("bước"); plt.ylabel("loss (TB trượt 10)"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("report/figures/loss_curves.png", dpi=110, bbox_inches="tight"); plt.show()
"""),
    md("## 7. Suy luận và chấm E1–E3"),
    code(r"""
for k in ("e1", "e2", "e3"):
    !python -m lab.run infer --ckpt work/{k} --out work/{k}.jsonl --split test
    !python -m lab.run score --name {k.upper()} --pred work/{k}.jsonl --run work/{k}/run.json --out report/evidence/{k}.json
"""),
    md("## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)"),
    code(r"""
# E4: biến data_size — 10% ảnh train, giữ nguyên mọi thứ khác của setup tốt nhất (100% = setup gốc).
BEST = "e1"   # đặt theo kết quả E1–E3
!python -m lab.run train --exp {BEST} --out work/e4_data_size_0.1 --train-fraction 0.1
!python -m lab.run infer --ckpt work/e4_data_size_0.1 --out work/e4_data_size_0.1.jsonl --split test
!python -m lab.run score --name E4_data_size_0.1 --pred work/e4_data_size_0.1.jsonl --run work/e4_data_size_0.1/run.json --out report/evidence/e4_data_size_0.1.json
!python -m lab.run score --name E4_data_size_1.0 --pred work/{BEST}.jsonl --run work/{BEST}/run.json --out report/evidence/e4_data_size_1.0.json
"""),
    code(r"""
# Bảng tổng hợp + ảnh minh hoạ đúng/sai (box xanh = đúng, đỏ = dự đoán)
from lab.viz import summary_table, make_gallery
print("| Mã | Tham số | VRAM | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole | valid | mIoU |")
print(summary_table("report/evidence"))
test = read_split("test")
for k in ("e0", BEST):
    caps = make_gallery(test, f"work/{k}.jsonl", "report/figures", k.upper())
    for c in caps: print(c["file"], c["tp"], c["fp"], c["fn"], c["gt"], c["pred"])
"""),
    code(r"""
# E5 (một phần): hàm reward IoU + test, không cần GPU
!python -m pytest -q lab/tests -k reward
!python scripts/check_submission.py
"""),
    code(r"""
# Gói kết quả nhỏ (không có trọng số) để tải về
!mkdir -p /content/out && cp -r report /content/out/ && cp work/*.jsonl /content/out/ 2>/dev/null; \
 for k in e1 e2 e3 e4_data_size_0.1; do mkdir -p /content/out/runs/$k; cp work/$k/run.json work/$k/train_log.json /content/out/runs/$k/ 2>/dev/null; done; \
 cd /content/out && zip -qr /content/day10_results.zip . && ls -la /content/day10_results.zip
from google.colab import files
files.download("/content/day10_results.zip")
"""),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = Path(__file__).resolve().parents[1] / "lab_day10.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
