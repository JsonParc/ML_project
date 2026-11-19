"""
발표 사이트(site/)에 결과를 넣는다.

- results/web/*.json  -> site/data/data.js (window.DATA = {...})
  JSON을 fetch로 읽지 않고 <script>로 넣는 이유: 인터넷이 안 될 때 index.html을 더블클릭해도(file://) 열리게.
- results/*.png       -> site/img/
- results/media/*     -> site/media/
실행: python scripts/06_build_site.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MEDIA, RESULTS, ROOT, WEB

SITE = ROOT / "site"


def main():
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    data = {}
    for name in ["reproduce", "ensemble", "decoding", "signals", "ml"]:
        f = WEB / f"{name}.json"
        if f.exists():
            data[name] = json.loads(f.read_text())
    rec = np.load(DATA / "recordings.npz", allow_pickle=False)
    data["eye"] = {"x": np.round(rec["stim_x"], 3).tolist(), "y": np.round(rec["stim_y"], 3).tolist()}
    meta = DATA / "recordings_meta.json"
    if meta.exists():
        data["recordings_meta"] = json.loads(meta.read_text())
    js = "window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (SITE / "data" / "data.js").write_text(js)
    for d, pat, src in [("img", "*.png", RESULTS), ("media", "*", MEDIA)]:
        (SITE / d).mkdir(exist_ok=True)
        for f in src.glob(pat):
            if f.is_file():
                shutil.copy2(f, SITE / d / f.name)
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site/ 갱신 완료: data.js {len(js) / 1e3:.0f} KB, 전체 {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
