import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import onnxruntime as ort
import spacemit_ort


MODEL = "/home/mjn/herbsecurity_ws/models/gouqi_yolo/gouqi_yolo_int8.onnx"


print("=" * 80)
print("ORT file:", ort.__file__)
print("ORT version:", getattr(ort, "__version__", "unknown"))
print("providers:", ort.get_available_providers())
print("=" * 80)

if "SpaceMITExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("没有 SpaceMITExecutionProvider，当前不是进迭加速版 ONNXRuntime")

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# 先降低图优化强度，避免优化后生成更大的加速子图
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

try:
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
except Exception as e:
    print("allow_spinning config failed:", e)

print("create session...")
sess = ort.InferenceSession(
    MODEL,
    sess_options=so,
    providers=["SpaceMITExecutionProvider"]
)

try:
    sess.disable_fallback()
except Exception as e:
    print("disable_fallback failed:", e)

print("session providers:", sess.get_providers())

inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]

print("input :", inp.name, inp.shape, inp.type)
print("output:", out.name, out.shape, out.type)

if "uint8" in inp.type:
    x = np.random.randint(0, 255, (1, 3, 320, 320), dtype=np.uint8)
else:
    x = np.random.rand(1, 3, 320, 320).astype(np.float32)

print("run once...")
t0 = time.time()
y = sess.run(None, {inp.name: x})
dt = (time.time() - t0) * 1000

print("OK")
print("output shape:", y[0].shape)
print("infer ms:", dt)