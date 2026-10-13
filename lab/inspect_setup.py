"""Kiểm tra bước 4–5 trong tiến trình riêng: số token ảnh, mask loss, số module LoRA, tham số từng khối."""
import gc
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.labels import get_prompt, objects_to_answer  # noqa: E402
from lab.vlm import (PROJ_KEYS, apply_setup, count_params, encode_example, load_model,  # noqa: E402
                     lora_regex, make_collate, matched_modules, open_image, read_split)


def main():
    model, processor = load_model()
    report = {"param_groups": count_params(model)}
    print("Tham số theo khối:", json.dumps(report["param_groups"], indent=1))

    # ---- bước 4: mã hoá + mask
    e = read_split("train")[0]
    ans = objects_to_answer(e["objects"])
    enc = encode_example(processor, open_image(e), get_prompt(), ans)
    img_tok = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((enc["input_ids"] == img_tok).sum())
    learned = processor.tokenizer.decode(enc["input_ids"][enc["labels"] != -100])
    print("keys:", list(enc.keys()))
    print("image_grid_thw:", enc["image_grid_thw"].tolist(), " số token ảnh:", n_img)
    print("seq len:", enc["input_ids"].shape[1], " token có loss:", int((enc["labels"] != -100).sum()))
    print("đáp án      :", ans)
    print("phần học    :", repr(learned))
    assert n_img == 220, n_img
    assert learned.startswith(ans) and "<|im_end|>" in learned
    batch = make_collate(processor.tokenizer.pad_token_id)([enc, encode_example(
        processor, open_image(read_split("train")[1]), get_prompt(),
        objects_to_answer(read_split("train")[1]["objects"]))])
    print("collate:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
    report.update({"image_tokens": n_img, "learned_text": learned})

    # ---- bước 5: module LoRA
    n_llm = len(matched_modules(model, lora_regex("llm")))
    n_all = len(matched_modules(model, lora_regex("llm+vit")))
    proj = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in PROJ_KEYS))
    print(f"LoRA modules E1/E2 = {n_llm}, E3 = {n_all}; projector params = {proj:,}")
    assert n_llm == 252 and n_all == 348, (n_llm, n_all)
    report.update({"lora_modules_llm": n_llm, "lora_modules_llm_vit": n_all, "projector_params": proj})
    names = [n for n, _ in model.named_modules()]
    print("vd tên module:", [n for n in names if re.search(r"(merger|blocks\.0\.|layers\.0\.)", n)][:20])
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for exp, targets, proj_on in (("e1", "llm", False), ("e2", "llm", True), ("e3", "llm+vit", True)):
        m, _ = load_model()
        _, info = apply_setup(m, targets, proj_on)
        report[f"trainable_{exp}"] = info["trainable_params"]
        del m, _
        gc.collect()
        torch.cuda.empty_cache()
    print("E2 − E1 =", f"{report['trainable_e2'] - report['trainable_e1']:,}")
    Path(ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "setup_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
