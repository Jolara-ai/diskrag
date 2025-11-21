#!/usr/bin/env bash
set -euo pipefail

# 環境設定
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_BIN="python3"

echo "[1/7] 建立/啟用 venv: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -V
pip -V

echo "[2/7] 升級 pip 並安裝依賴"
pip install -U pip wheel setuptools
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
  pip install -r "$PROJECT_ROOT/requirements.txt"
fi
pip install Cython numpy scikit-learn numba tqdm

echo "[3/7] 編譯 pydiskann Cython 擴充"
cd "$PROJECT_ROOT"
python pydiskann/setup.py build_ext --inplace

echo "[4/7] 確認擴充模組可載入"
python - <<'PY'
import importlib
m = importlib.import_module('pydiskann.cython_utils')
print('Loaded:', m.__name__)
print('Has funcs:', hasattr(m, 'l2_distance_fast_cython'), hasattr(m, 'greedy_search_cython'))
PY

echo "[5/7] 功能正確性測試 (數值一致性)"
python - <<'PY'
import numpy as np
from pydiskann.cython_utils import l2_distance_fast_cython, cosine_similarity_cython

rs = np.random.RandomState(0)
x = rs.randn(128).astype(np.float32)
y = rs.randn(128).astype(np.float32)

l2_sq_np = float(np.sum((x-y)**2))
l2_sq_cy = float(l2_distance_fast_cython(x, y))
cos_np = float(1.0 - (np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))))
cos_cy = float(cosine_similarity_cython(x, y))

print('L2^2 np:', l2_sq_np, ' cy:', l2_sq_cy, ' close:', np.allclose(l2_sq_np, l2_sq_cy, rtol=1e-5, atol=1e-6))
print('Cos  np:', cos_np,   ' cy:', cos_cy,   ' close:', np.allclose(cos_np,   cos_cy,   rtol=1e-5, atol=1e-6))

assert np.allclose(l2_sq_np, l2_sq_cy, rtol=1e-5, atol=1e-6)
assert np.allclose(cos_np,   cos_cy,   rtol=1e-5, atol=1e-6)
print('數值正確性：通過')
PY

echo "[6/7] 整合測試 (建圖 + 搜尋)"
python - <<'PY'
import numpy as np
from pydiskann.vamana_graph import build_vamana_with_pq, beam_search_with_pq
from pydiskann.pq.fast_pq import DiskANNPQ

rs = np.random.RandomState(42)
N, D = 2000, 64
vectors = rs.randn(N, D).astype(np.float32)

pq = DiskANNPQ(n_subvectors=8, n_centroids=256)
pq.fit(vectors)

graph = build_vamana_with_pq(
    points=vectors, pq_model=pq, R=16, L=32, alpha=1.2,
    use_pq_in_build=False, show_progress=False, distance_metric='l2'
)
graph.enable_pq_search(True)

q = rs.randn(D).astype(np.float32)
res = beam_search_with_pq(graph, q, start_idx=None, beam_width=8, k=5, use_pq=True)
print('Top-5 (dist, idx):', res)
assert isinstance(res, list) and len(res) > 0
print('整合測試：通過')
PY

echo "[7/7] 微基準 (距離計算迭代)"
python - <<'PY'
import time
import numpy as np
from pydiskann.cython_utils import l2_distance_fast_cython as l2_cy

rs = np.random.RandomState(123)
X = rs.randn(2000, 128).astype(np.float32)
Y = rs.randn(2000, 128).astype(np.float32)

t0 = time.time()
s = 0.0
for i in range(2000):
    s += float(l2_cy(X[i], Y[i]))
t1 = time.time()
print('Cython l2^2 time(s):', round(t1-t0, 4), ' sum:', round(s, 3))

t0 = time.time()
s2 = 0.0
for i in range(2000):
    z = X[i]-Y[i]
    s2 += float(np.dot(z, z))
t1 = time.time()
print('NumPy per-sample time(s):', round(t1-t0, 4), ' sum:', round(s2, 3))
PY

echo "完成：Cython 模組測試執行成功。"


