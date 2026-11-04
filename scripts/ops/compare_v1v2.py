"""Compare CLIP content distribution: v1 gt sample vs v2 accumulated tiles."""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/tmp/kilo")
from balance_val import classify

v1_files = sorted(Path("/media/phhofm/SSD/pd12m_v1/pd12m_train_full").glob("*.png"))
rng = random.Random(0)
v1_s = [v1_files[i] for i in rng.sample(range(len(v1_files)), 500)]
v2_files = sorted(Path("/media/phhofm/SSD/pd12m_v2/workspace/scored").glob("*.png"))[:800]
print(f"v1 sample {len(v1_s)}, v2 sample {len(v2_files)}")
c1 = Counter(classify(v1_s))
c2 = Counter(classify(v2_files))
print(f"{'category':34s} {'v1%':>6s} {'v2%':>6s}")
for cat in sorted(set(c1) | set(c2), key=lambda c: -(c1.get(c, 0) + c2.get(c, 0))):
    print(f"{cat[:32]:34s} {100*c1.get(cat,0)/len(v1_s):6.1f} {100*c2.get(cat,0)/max(len(v2_files),1):6.1f}")
