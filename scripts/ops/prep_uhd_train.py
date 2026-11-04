"""Prepare UHD-IQA training split as whole-scene 512 crops."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

SRC = Path("/media/phhofm/SSD/uhdiqa/training/training")
OUT = Path("/media/phhofm/SSD/uhd_ws_train/input")


def conv(p: str) -> str:
    p = Path(p)
    dst = OUT / f"train_{p.stem}.png"
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = 512 / min(w, h)
        r = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.BICUBIC)
        W, H = r.size
        r.crop(((W - 512) // 2, (H - 512) // 2,
                (W - 512) // 2 + 512, (H - 512) // 2 + 512)).save(dst)
    return dst.name


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.jpg"))
    with ProcessPoolExecutor(max_workers=8) as pool:
        names = list(pool.map(conv, files, chunksize=32))
    print("prepared:", len(names))
