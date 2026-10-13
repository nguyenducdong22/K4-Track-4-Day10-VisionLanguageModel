"""CLI cho từng thí nghiệm, mỗi lệnh chạy trong một tiến trình riêng (VRAM đỉnh đo sạch).

  python -m lab.run infer --out work/e0.jsonl                       # E0 zero-shot
  python -m lab.run train --exp e1 --out work/e1                    # huấn luyện
  python -m lab.run infer --ckpt work/e1 --out work/e1.jsonl        # suy luận với adapter
  python -m lab.run score --name E1 --pred work/e1.jsonl --run work/e1/run.json --out report/evidence/e1.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cfg(exp):
    return yaml.safe_load((ROOT / "configs" / f"{exp}.yaml").read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--exp", required=True, help="e1 | e2 | e3")
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=int)
    t.add_argument("--train-fraction", type=float, default=1.0)
    t.add_argument("--coord", default="norm1000")
    t.add_argument("--max-steps", type=int)
    t.add_argument("--split", default="train")

    i = sub.add_parser("infer")
    i.add_argument("--ckpt", help="thư mục chứa adapter/ (+ projector.pt); bỏ trống = zero-shot")
    i.add_argument("--out", required=True)
    i.add_argument("--split", default="test")
    i.add_argument("--limit", type=int)
    i.add_argument("--coord", default="norm1000")
    i.add_argument("--batch-size", type=int, default=8)
    i.add_argument("--max-new-tokens", type=int, default=256)

    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--run", help="run.json để lấy meta")
    s.add_argument("--split", default="test")
    s.add_argument("--coord", default="norm1000")

    a = ap.parse_args(argv)

    if a.cmd == "train":
        from lab.vlm import train
        cfg = load_cfg(a.exp)
        if a.rank:
            cfg["lora"]["r"], cfg["lora"]["alpha"] = a.rank, 2 * a.rank
        train(cfg, a.out, train_fraction=a.train_fraction, coord=a.coord,
              max_steps=a.max_steps, split=a.split)
    elif a.cmd == "infer":
        from lab.vlm import infer, load_model, load_trained, read_split
        model, proc = load_trained(a.ckpt) if a.ckpt else load_model()
        entries = read_split(a.split)[: a.limit] if a.limit else read_split(a.split)
        infer(model, proc, entries, a.out, coord=a.coord, batch_size=a.batch_size,
              max_new_tokens=a.max_new_tokens)
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", a.pred, "--name",
               a.name, "--out", a.out, "--split", a.split, "--coord", a.coord,
               "--splits-dir", str(ROOT / "data" / "splits")]
        if a.run:
            run = json.loads(Path(a.run).read_text(encoding="utf-8"))
            for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
                cmd += ["--meta", f"{k}={run[k]}"]
            for k in ("train_fraction", "steps", "train_images"):
                if k in run:
                    cmd += ["--meta", f"{k}={run[k]}"]
            if run["cfg"].get("lora"):
                cmd += ["--meta", f"lora_r={run['cfg']['lora']['r']}"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
