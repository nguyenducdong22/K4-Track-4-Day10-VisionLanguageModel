"""Bước 3–7: nạp Qwen3-VL-4B 4-bit, mã hoá ví dụ + mask loss, setup LoRA/đóng băng, train, suy luận."""
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from lab.labels import get_prompt, objects_to_answer

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LLM_REGEX = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_REGEX = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJ_KEYS = ("visual.merger", "visual.deepstack_merger_list")


# ----------------------------------------------------------------------------- model
def skip_modules(model_id=MODEL_ID):
    """Tên ĐẦY ĐỦ các Linear của ViT + projector để bitsandbytes KHÔNG lượng tử hoá.

    Với transformers 5.19, llm_int8_skip_modules=["visual", "lm_head"] chỉ khớp được lm_head:
    các Linear trong model.visual.* vẫn bị đổi thành Linear4bit (đã kiểm chứng: merger.linear_fc1
    là Linear4bit uint8, projector chỉ còn 54,6M "tham số" thay vì 109M). Vì vậy liệt kê tường minh.
    """
    from transformers import AutoConfig
    vc = AutoConfig.from_pretrained(model_id).vision_config
    names = ["visual", "lm_head"]
    for i in range(vc.depth):
        names += [f"model.visual.blocks.{i}.{s}"
                  for s in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")]
    names += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
    for i in range(len(vc.deepstack_visual_indexes)):
        names += [f"model.visual.deepstack_merger_list.{i}.linear_fc{j}" for j in (1, 2)]
    return names


def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                           llm_int8_skip_modules=skip_modules(model_id))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=q, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def count_params(model):
    groups = {"visual.blocks(ViT)": 0, "projector": 0, "language_model+embed": 0, "lm_head": 0,
              "other": 0}
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        k = p.numel()
        # tham số 4-bit của bitsandbytes được nén 2 giá trị / byte
        if p.__class__.__name__ == "Params4bit":
            k *= 2
        if any(s in n for s in PROJ_KEYS):
            groups["projector"] += k
        elif "visual" in n:
            groups["visual.blocks(ViT)"] += k
        elif "language_model" in n:
            groups["language_model+embed"] += k
        elif "lm_head" in n:
            groups["lm_head"] += k
        else:
            groups["other"] += k
    groups["total"] = sum(groups.values())
    return groups


# ----------------------------------------------------------------------------- data
def read_split(split):
    rows = []
    with open(DATA / "splits" / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def open_image(entry):
    return Image.open(DATA / entry["image_path"]).convert("RGB")


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    """Chuỗi đầy đủ (prompt + đáp án) với labels = -100 trên prompt và token ảnh."""
    user = user_msg(image, prompt)
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kw)
    n = p_enc["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n], p_enc["input_ids"][0]), "prompt không là tiền tố"
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    # bỏ '\n' sau <|im_end|> cuối (template tự thêm), chỉ học đáp án + <|im_end|>
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = full["input_ids"][0]
    last = (ids == im_end).nonzero().flatten()
    if len(last) and last[-1] >= n:
        labels[0, last[-1] + 1:] = -100
    out = dict(full)
    out["labels"] = labels
    return out


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, processor, entries, coord="norm1000"):
        self.p, self.e, self.coord = processor, entries, coord

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        e = self.e[i]
        ans = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.p, open_image(e), get_prompt(self.coord, e["width"], e["height"]),
                              ans)


def make_collate(pad_id):
    pad_vals = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}

    def collate(batch):
        out = {}
        L = max(b["input_ids"].shape[1] for b in batch)
        for k in batch[0]:
            if k in ("pixel_values", "image_grid_thw"):
                out[k] = torch.cat([b[k] for b in batch], 0)  # nối, KHÔNG pad
            elif torch.is_tensor(batch[0][k]) and batch[0][k].dim() == 2 \
                    and batch[0][k].shape[1] == batch[0]["input_ids"].shape[1]:
                pv = pad_vals.get(k, 0)
                rows = []
                for b in batch:
                    t = b[k]
                    pad = torch.full((1, L - t.shape[1]), pv, dtype=t.dtype)
                    rows.append(torch.cat([t, pad], 1))  # đệm phải khi train
                out[k] = torch.cat(rows, 0)
            else:
                out[k] = torch.cat([b[k] for b in batch], 0) if torch.is_tensor(batch[0][k]) \
                    else [b[k] for b in batch]
        return out

    return collate


# ----------------------------------------------------------------------------- setup
def lora_regex(targets):
    return LLM_REGEX if targets == "llm" else f"({LLM_REGEX})|({VIT_REGEX})"


def matched_modules(model, regex):
    return [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    regex = lora_regex(targets)
    mods = matched_modules(model, regex)
    print(f"LoRA target modules: {len(mods)}  (vd {mods[0]} … {mods[-1]})")
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                          target_modules=regex, bias="none"))
    extra = []
    if train_projector:
        for n, p in pm.named_parameters():
            if any(k in n for k in PROJ_KEYS) and "lora_" not in n:
                p.data = p.data.float()
                p.requires_grad = True
                extra.append(n)
    for n, p in pm.named_parameters():  # GradScaler cần tham số huấn luyện ở fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n)
    print(f"trainable params: {n_tr:,}  (LoRA {n_lora:,} + projector {n_tr - n_lora:,})")
    return pm, {"n_lora_modules": len(mods), "trainable_params": n_tr, "lora_params": n_lora,
                "projector_params": n_tr - n_lora, "extra_trainable": extra}


# ----------------------------------------------------------------------------- train
def train(cfg, out_dir, train_fraction=1.0, coord="norm1000", max_steps=None, split="train",
          log_every=10, num_workers=2):
    from transformers import get_cosine_schedule_with_warmup
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model, processor = load_model(cfg.get("model_id", MODEL_ID))
    lora = cfg["lora"]
    pm, info = apply_setup(model, lora["targets"], cfg["train_projector"], lora["r"],
                           lora["alpha"], lora["dropout"])
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.config.use_cache = False
    pm.train()

    entries = read_split(split)
    if train_fraction < 1.0:
        rng = random.Random(seed)
        entries = rng.sample(entries, max(1, int(round(len(entries) * train_fraction))))
    print(f"train images: {len(entries)}")
    ds = TrainSet(processor, entries, coord)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=g,
                                     collate_fn=make_collate(processor.tokenizer.pad_token_id),
                                     num_workers=num_workers, drop_last=True,
                                     persistent_workers=num_workers > 0)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps = max_steps or cfg["max_steps"]
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda")
    accum = cfg["grad_accum"]

    log, step, it = [], 0, iter(dl)
    while step < steps:
        tot = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = pm(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn fp16?)")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        log.append({"step": step, "loss": tot, "lr": sched.get_last_lr()[0],
                    "grad_norm": float(gn), "min": (time.time() - t0) / 60})
        if step % log_every == 0 or step == 1:
            print(f"step {step:4d}/{steps} loss {tot:.4f} lr {log[-1]['lr']:.2e} "
                  f"gn {float(gn):.2f} {log[-1]['min']:.1f} min "
                  f"vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        if step % 100 == 0 or step == steps:
            save_trained(pm, out_dir, info["extra_trainable"])

    train_min = (time.time() - t0) / 60
    run = {"trainable_params": info["trainable_params"], "lora_params": info["lora_params"],
           "projector_params": info["projector_params"], "n_lora_modules": info["n_lora_modules"],
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round(train_min, 1), "steps": steps, "train_images": len(entries),
           "train_fraction": train_fraction, "coord": coord, "cfg": cfg}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    print(json.dumps({k: v for k, v in run.items() if k != "cfg"}, indent=2))
    return run


def save_trained(pm, out_dir, extra_names):
    out_dir = Path(out_dir)
    pm.save_pretrained(out_dir / "adapter")  # CHỈ lưu LoRA
    if extra_names:  # projector huấn luyện đầy đủ: lưu riêng, nếu không E2/E3 sẽ giống E1
        sd = {n: p.detach().cpu() for n, p in pm.named_parameters() if n in set(extra_names)}
        torch.save(sd, out_dir / "projector.pt")


def load_trained(out_dir):
    from peft import PeftModel
    out_dir = Path(out_dir)
    model, processor = load_model()
    pm = PeftModel.from_pretrained(model, out_dir / "adapter")
    pp = out_dir / "projector.pt"
    if pp.exists():
        sd = torch.load(pp, map_location="cpu")
        n_loaded = 0
        for n, p in pm.named_parameters():
            if n in sd:
                p.data.copy_(sd[n].to(p.device, p.dtype))
                n_loaded += 1
        assert n_loaded == len(sd), (n_loaded, len(sd))
        print(f"đã nạp lại {n_loaded} tensor projector")
    pm.eval()
    return pm, processor


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def infer(model, processor, entries, out_path, coord="norm1000", batch_size=8, max_new_tokens=256):
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():  # chạy tiếp nếu bị ngắt
        done = {json.loads(l)["image_id"] for l in out_path.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    todo = [e for e in entries if e["image_id"] not in done]
    print(f"infer {len(todo)} ảnh (đã có {len(done)})")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            convs = [[user_msg(open_image(e), get_prompt(coord, e["width"], e["height"]))]
                     for e in chunk]
            inputs = processor.apply_chat_template(
                convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // batch_size) % 5 == 0:
                print(f"  {i + len(chunk)}/{len(todo)}  {(time.time() - t0) / 60:.1f} min",
                      flush=True)
    print(f"xong infer: {(time.time() - t0) / 60:.1f} min")
