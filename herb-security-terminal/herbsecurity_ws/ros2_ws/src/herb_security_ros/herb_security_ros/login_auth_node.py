#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
login_auth_node.py

K1 / MUSE Pi Pro 人脸登录认证 ROS2 节点

修复点：
1. 系统 Python 启动 ROS2 console_scripts 时找不到虚拟环境 onnxruntime，因此手动加入 venv site-packages。
2. 没有人脸 NoFace / AlignFail 不再当作认证失败。
3. 识别成功需要连续确认 require_pass_frames 次，避免旧帧误成功。
4. latest_result 增加 result_time，旧框只保留 result_hold_time 秒。
5. 摄像头画面始终发布给 UI；只有认证激活时才进入推理队列。
6. MJN 固定为管理员权限，其余识别通过人员为普通用户。
"""

import sys
import site
from pathlib import Path

# ============================================================
# 让 /usr/bin/python3 启动的 ROS2 节点也能找到虚拟环境库
# ============================================================

VENV_SITE = Path.home() / "venvs" / "ROS_herb_robot" / "lib" / "python3.12" / "site-packages"

if VENV_SITE.exists():
    site.addsitedir(str(VENV_SITE))
    sys.path.insert(0, str(VENV_SITE))

import cv2
import time
import json
import queue
import threading
import traceback
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)


# ============================================================
# 默认路径
# ============================================================

WORK_DIR = Path.home() / "herbsecurity_ws"

DEFAULT_DET_MODEL = WORK_DIR / "models" / "buffalo_s" / "det_500m.onnx"
DEFAULT_REC_MODEL = WORK_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"
DEFAULT_DB_PATH = WORK_DIR / "face_data" / "face_database.npz"


# ============================================================
# 默认参数
# ============================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480

DET_SIZE = (320, 320)
DET_THRESH = 0.45
NMS_THRESH = 0.40

SIM_THRESHOLD = 0.35

FRAME_QUEUE_SIZE = 1

ADMIN_NAMES = {"MJN", "mjn", "Mjn"}

ARC_FACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

ANCHOR_CACHE = {}

SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# ============================================================
# 工具函数
# ============================================================

def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def json_loads_safe(text: str) -> dict:
    if text is None:
        return {}

    text = str(text).strip()

    if not text:
        return {}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"data": obj}
    except Exception:
        return {"raw": text}


def l2_normalize(vec):
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


def get_role_by_name(name: str) -> str:
    if name in ADMIN_NAMES:
        return "admin"

    return "user"


def role_to_cn(role: str) -> str:
    if role == "admin":
        return "管理员"

    if role == "user":
        return "普通用户"

    return "--"


# ============================================================
# ONNXRuntime
# ============================================================

def create_ort_session(model_path: Path, threads: int = 4):
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    return session


def print_model_info(name, session):
    print(f"\n{name} 模型信息:")

    print("  Inputs:")
    for item in session.get_inputs():
        print(f"    {item.name} | shape={item.shape} | type={item.type}")

    print("  Outputs:")
    for item in session.get_outputs():
        print(f"    {item.name} | shape={item.shape} | type={item.type}")

    print("  Providers:", session.get_providers())


def load_face_database(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"未找到人脸数据库: {db_path}")

    try:
        db = np.load(str(db_path), allow_pickle=False)
    except ValueError:
        print("警告：face_database.npz 含 object 数据，改用 allow_pickle=True 加载。")
        db = np.load(str(db_path), allow_pickle=True)

    if "names" not in db or "embeddings" not in db:
        raise RuntimeError("face_database.npz 必须包含 names 和 embeddings 两个字段。")

    names_raw = db["names"]
    embeddings = db["embeddings"].astype(np.float32)

    names = []
    for name in names_raw:
        if isinstance(name, bytes):
            names.append(name.decode("utf-8"))
        else:
            names.append(str(name))

    embeddings = np.array(
        [l2_normalize(e) for e in embeddings],
        dtype=np.float32,
    )

    print("\n已加载人脸数据库:")
    for name in names:
        role = get_role_by_name(name)
        print(f"  - {name} | {role_to_cn(role)}")

    print("数据库特征维度:", embeddings.shape)

    return np.array(names), embeddings


def search_face(embedding, names, db_embeddings, threshold):
    embedding = l2_normalize(embedding)

    sims = np.dot(db_embeddings, embedding)

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = str(names[best_idx])

    if best_score >= threshold:
        return best_name, best_score, True

    return "Unknown", best_score, False


# ============================================================
# SCRFD 后处理
# ============================================================

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []

    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)

    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def preprocess_det(img, input_size):
    input_w, input_h = input_size
    img_h, img_w = img.shape[:2]

    im_ratio = img_h / img_w
    model_ratio = input_h / input_w

    if im_ratio > model_ratio:
        new_h = input_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = input_w
        new_h = int(new_w * im_ratio)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    det_scale = new_h / img_h

    resized_img = cv2.resize(img, (new_w, new_h))

    det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized_img

    blob = cv2.dnn.blobFromImage(
        det_img,
        scalefactor=1.0 / 128.0,
        size=(input_w, input_h),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )

    return blob.astype(np.float32), det_scale


def get_anchor_centers(input_size, stride, num_anchors):
    key = (input_size[0], input_size[1], stride, num_anchors)

    if key in ANCHOR_CACHE:
        return ANCHOR_CACHE[key]

    input_w, input_h = input_size
    feat_w = input_w // stride
    feat_h = input_h // stride

    y, x = np.mgrid[:feat_h, :feat_w]

    anchor_centers = np.stack((x, y), axis=-1).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))

    if num_anchors > 1:
        anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1)
        anchor_centers = anchor_centers.reshape((-1, 2))

    ANCHOR_CACHE[key] = anchor_centers

    return anchor_centers


def detect_faces(det_sess, det_input_name, img):
    input_size = DET_SIZE

    blob, det_scale = preprocess_det(img, input_size)

    outputs = det_sess.run(None, {det_input_name: blob})

    if len(outputs) != 9:
        print("错误：检测模型输出数量不是 9，实际为:", len(outputs))
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

    strides = [8, 16, 32]

    scores_list = []
    bboxes_list = []
    kpss_list = []

    for idx, stride in enumerate(strides):
        scores = outputs[idx].reshape(-1)
        bbox_preds = outputs[idx + 3].reshape((-1, 4))
        kps_preds = outputs[idx + 6].reshape((-1, 10))

        input_w, input_h = input_size
        feat_w = input_w // stride
        feat_h = input_h // stride

        num_anchors = int(scores.shape[0] / (feat_w * feat_h))
        anchor_centers = get_anchor_centers(input_size, stride, num_anchors)

        min_len = min(
            scores.shape[0],
            bbox_preds.shape[0],
            kps_preds.shape[0],
            anchor_centers.shape[0],
        )

        scores = scores[:min_len]
        bbox_preds = bbox_preds[:min_len]
        kps_preds = kps_preds[:min_len]
        anchor_centers = anchor_centers[:min_len]

        pos_inds = np.where(scores >= DET_THRESH)[0]

        if len(pos_inds) == 0:
            continue

        bbox_preds = bbox_preds * stride
        kps_preds = kps_preds * stride

        bboxes = distance2bbox(anchor_centers, bbox_preds)
        kpss = distance2kps(anchor_centers, kps_preds)

        scores_list.append(scores[pos_inds])
        bboxes_list.append(bboxes[pos_inds])
        kpss_list.append(kpss[pos_inds])

    if len(scores_list) == 0:
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

    scores = np.concatenate(scores_list, axis=0)
    bboxes = np.concatenate(bboxes_list, axis=0)
    kpss = np.concatenate(kpss_list, axis=0)

    bboxes = bboxes / det_scale
    kpss = kpss / det_scale
    kpss = kpss.reshape((-1, 5, 2))

    img_h, img_w = img.shape[:2]

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_w - 1)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_h - 1)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img_w - 1)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img_h - 1)

    dets = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32)

    order = dets[:, 4].argsort()[::-1]
    dets = dets[order]
    kpss = kpss[order]

    keep = nms(dets, NMS_THRESH)

    return dets[keep], kpss[keep]


def get_largest_face(dets, kpss):
    if dets.shape[0] == 0:
        return None, None

    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    idx = int(np.argmax(areas))

    return dets[idx], kpss[idx]


# ============================================================
# ArcFace
# ============================================================

def norm_crop(img, landmark):
    landmark = landmark.astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(
        landmark,
        ARC_FACE_DST,
        method=cv2.LMEDS,
    )

    if M is None:
        return None

    aligned = cv2.warpAffine(
        img,
        M,
        (112, 112),
        borderValue=0.0,
    )

    return aligned


def get_face_embedding(rec_sess, rec_input_name, aligned_face):
    blob = cv2.dnn.blobFromImage(
        aligned_face,
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )

    blob = blob.astype(np.float32)

    outputs = rec_sess.run(None, {rec_input_name: blob})

    embedding = outputs[0].reshape(-1).astype(np.float32)

    return l2_normalize(embedding)


def run_face_inference(
    frame,
    det_sess,
    det_input_name,
    rec_sess,
    rec_input_name,
    names,
    db_embeddings,
    threshold,
):
    start_time = time.time()

    dets, kpss = detect_faces(det_sess, det_input_name, frame)
    det, kps = get_largest_face(dets, kpss)

    if det is None:
        return {
            "name": "NoFace",
            "role": "--",
            "score": 0.0,
            "passed": False,
            "bbox": None,
            "infer_ms": (time.time() - start_time) * 1000.0,
            "status": "noface",
            "error": None,
        }

    aligned = norm_crop(frame, kps)

    if aligned is None:
        return {
            "name": "AlignFail",
            "role": "--",
            "score": 0.0,
            "passed": False,
            "bbox": tuple(det[:4].astype(int).tolist()),
            "infer_ms": (time.time() - start_time) * 1000.0,
            "status": "align_fail",
            "error": None,
        }

    embedding = get_face_embedding(rec_sess, rec_input_name, aligned)

    name, score, passed = search_face(
        embedding,
        names,
        db_embeddings,
        threshold,
    )

    role = get_role_by_name(name) if passed else "--"

    return {
        "name": name,
        "role": role,
        "score": score,
        "passed": passed,
        "bbox": tuple(det[:4].astype(int).tolist()),
        "infer_ms": (time.time() - start_time) * 1000.0,
        "status": "passed" if passed else "failed",
        "error": None,
    }


# ============================================================
# 摄像头与队列
# ============================================================

def open_camera(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败: /dev/video{cam_index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError(f"摄像头能打开但读取失败: /dev/video{cam_index}")

    print(f"摄像头打开成功: /dev/video{cam_index}")
    print(f"首帧尺寸: {frame.shape}")

    return cap


def push_latest_frame(frame_queue, frame):
    try:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put_nowait(frame)

    except queue.Full:
        pass


def send_stop_sentinel(frame_queue):
    try:
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put_nowait(None)

    except queue.Full:
        pass


def draw_result(frame, result):
    if result is None:
        return frame

    bbox = result.get("bbox", None)
    name = str(result.get("name", "--"))
    role = str(result.get("role", "--"))
    score = float(result.get("score", 0.0))
    passed = bool(result.get("passed", False))
    status = str(result.get("status", ""))
    infer_ms = float(result.get("infer_ms", 0.0))

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        color = (0, 220, 0) if passed else (0, 0, 255)

        if status == "verifying":
            color = (0, 180, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{name} {score:.3f}"

        if passed:
            text += f" {role}"

        cv2.putText(
            frame,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

    if status == "noface":
        show = "NoFace"
    elif status == "verifying":
        show = "VERIFYING"
    elif passed:
        show = "PASS"
    else:
        show = name

    cv2.putText(
        frame,
        f"FaceAuth: {show} | score={score:.3f} | {infer_ms:.1f}ms",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    return frame


# ============================================================
# LoginAuthNode
# ============================================================

class LoginAuthNode(Node):
    def __init__(self):
        super().__init__("login_auth_node")

        self.declare_parameter("cam_index", 20)
        self.declare_parameter("threshold", SIM_THRESHOLD)
        self.declare_parameter("infer_interval", 0.20)
        self.declare_parameter("frame_fps", 12.0)
        self.declare_parameter("result_interval", 1.0)
        self.declare_parameter("require_pass_frames", 2)
        self.declare_parameter("result_hold_time", 1.0)
        self.declare_parameter("threads", 4)
        self.declare_parameter("det_model", str(DEFAULT_DET_MODEL))
        self.declare_parameter("rec_model", str(DEFAULT_REC_MODEL))
        self.declare_parameter("db_path", str(DEFAULT_DB_PATH))

        self.cam_index = int(self.get_parameter("cam_index").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.infer_interval = float(self.get_parameter("infer_interval").value)
        self.frame_pub_fps = float(self.get_parameter("frame_fps").value)
        self.result_pub_interval = float(self.get_parameter("result_interval").value)
        self.require_pass_frames = int(self.get_parameter("require_pass_frames").value)
        self.result_hold_time = float(self.get_parameter("result_hold_time").value)
        self.threads = int(self.get_parameter("threads").value)

        self.det_model = Path(str(self.get_parameter("det_model").value)).expanduser()
        self.rec_model = Path(str(self.get_parameter("rec_model").value)).expanduser()
        self.db_path = Path(str(self.get_parameter("db_path").value)).expanduser()

        self.auth_active = False
        self.auth_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

        self.result_lock = threading.Lock()
        self.latest_result = None

        self.last_result_publish_time = 0.0
        self.last_result_key = ""
        self.last_frame_publish_time = 0.0

        self.pass_candidate_name = ""
        self.pass_confirm_count = 0

        self.cap = None
        self.infer_thread = None

        self.det_sess = None
        self.rec_sess = None
        self.det_input_name = None
        self.rec_input_name = None

        self.names = None
        self.db_embeddings = None

        self.face_result_pub = self.create_publisher(String, "/auth/face_result", 10)

        self.face_frame_pub = self.create_publisher(
            CompressedImage,
            "/auth/face_frame/compressed",
            SENSOR_QOS,
        )

        self.system_event_pub = self.create_publisher(String, "/system/event", 10)
        self.device_status_pub = self.create_publisher(String, "/device/status", 10)

        self.create_subscription(String, "/auth/request", self.on_auth_request, 10)
        self.create_subscription(String, "/auth/logout", self.on_auth_logout, 10)

        self.init_models()
        self.init_camera()
        self.start_inference_thread()

        self.capture_timer = self.create_timer(0.03, self.on_capture_timer)
        self.status_timer = self.create_timer(2.0, self.publish_camera_status)

        self.publish_system_event("login_auth_node 已启动", "info")

    # =====================================================
    # 初始化
    # =====================================================

    def init_models(self):
        self.get_logger().info("加载人脸数据库...")
        self.names, self.db_embeddings = load_face_database(self.db_path)

        self.get_logger().info("加载检测模型...")
        self.det_sess = create_ort_session(self.det_model, threads=self.threads)
        self.det_input_name = self.det_sess.get_inputs()[0].name
        print_model_info("检测", self.det_sess)

        self.get_logger().info("加载识别模型...")
        self.rec_sess = create_ort_session(self.rec_model, threads=self.threads)
        self.rec_input_name = self.rec_sess.get_inputs()[0].name
        print_model_info("识别", self.rec_sess)

    def init_camera(self):
        self.cap = open_camera(self.cam_index)
        self.publish_camera_status()

    def start_inference_thread(self):
        self.infer_thread = threading.Thread(
            target=self.inference_loop,
            daemon=True,
        )
        self.infer_thread.start()

    # =====================================================
    # 状态清理
    # =====================================================

    def reset_auth_result_state(self):
        with self.result_lock:
            self.latest_result = None

        self.last_result_publish_time = 0.0
        self.last_result_key = ""

        self.pass_candidate_name = ""
        self.pass_confirm_count = 0

    # =====================================================
    # 订阅回调
    # =====================================================

    def on_auth_request(self, msg: String):
        data = json_loads_safe(msg.data)
        cmd = str(data.get("cmd", data.get("raw", "start"))).strip()

        if cmd not in ["start", "auth", "login", "开始认证"]:
            return

        with self.auth_lock:
            self.auth_active = True

        self.reset_auth_result_state()

        self.publish_system_event("收到登录认证请求，开始人脸识别", "info")
        self.get_logger().info("收到 /auth/request，开始认证。")

    def on_auth_logout(self, msg: String):
        with self.auth_lock:
            self.auth_active = False

        self.reset_auth_result_state()

        self.publish_system_event("收到注销请求，认证状态已清除", "info")
        self.get_logger().info("收到 /auth/logout，认证状态已清除。")

    # =====================================================
    # 摄像头采集与画面发布
    # =====================================================

    def on_capture_timer(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.publish_system_event("人脸摄像头读取失败", "error")
            return

        with self.auth_lock:
            active = self.auth_active

        # 只有认证激活时才进入推理队列
        if active:
            push_latest_frame(self.frame_queue, frame.copy())

        # 画面始终发布给 UI
        now = time.time()

        if now - self.last_frame_publish_time >= 1.0 / max(1.0, self.frame_pub_fps):
            self.last_frame_publish_time = now

            with self.result_lock:
                if self.latest_result is not None:
                    result_age = now - float(self.latest_result.get("result_time", 0.0))

                    if result_age <= self.result_hold_time:
                        result_copy = dict(self.latest_result)
                    else:
                        result_copy = None
                else:
                    result_copy = None

            display_frame = frame.copy()
            display_frame = draw_result(display_frame, result_copy)

            self.publish_compressed_frame(display_frame)

    def publish_compressed_frame(self, frame):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )

        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "face_camera"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()

        self.face_frame_pub.publish(msg)

    # =====================================================
    # 推理线程
    # =====================================================

    def inference_loop(self):
        last_infer_end = 0.0

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                break

            with self.auth_lock:
                active = self.auth_active

            if not active:
                continue

            now = time.time()
            wait_time = self.infer_interval - (now - last_infer_end)

            if wait_time > 0:
                if self.stop_event.wait(wait_time):
                    break

            try:
                result = run_face_inference(
                    frame,
                    self.det_sess,
                    self.det_input_name,
                    self.rec_sess,
                    self.rec_input_name,
                    self.names,
                    self.db_embeddings,
                    self.threshold,
                )

            except Exception as e:
                result = {
                    "name": "Error",
                    "role": "--",
                    "score": 0.0,
                    "passed": False,
                    "bbox": None,
                    "infer_ms": 0.0,
                    "status": "error",
                    "error": repr(e),
                }

                print("\n后台推理异常:")
                traceback.print_exc()

            last_infer_end = time.time()

            self.handle_inference_result(result)

        print("登录认证推理线程已退出。")

    def handle_inference_result(self, result: dict):
        result["result_time"] = time.time()

        name = str(result.get("name", "--"))
        role = str(result.get("role", "--"))
        score = float(result.get("score", 0.0))
        passed = bool(result.get("passed", False))
        infer_ms = float(result.get("infer_ms", 0.0))
        status = str(result.get("status", ""))
        error = result.get("error", None)

        with self.result_lock:
            self.latest_result = result

        if error:
            self.get_logger().error(f"推理异常：{error}")
            self.publish_face_result(result)
            return

        # 没有人脸：不是认证失败
        if name in ["NoFace", "AlignFail"] or status in ["noface", "align_fail"]:
            self.pass_candidate_name = ""
            self.pass_confirm_count = 0

            now = time.time()
            if now - self.last_result_publish_time >= self.result_pub_interval:
                self.last_result_publish_time = now
                self.publish_face_result(result)

            return

        # 检测到人脸但未通过阈值：这才是失败
        if not passed:
            self.pass_candidate_name = ""
            self.pass_confirm_count = 0

            now = time.time()
            key = f"{name}:{score:.2f}"

            if (
                key != self.last_result_key
                or now - self.last_result_publish_time >= self.result_pub_interval
            ):
                self.last_result_key = key
                self.last_result_publish_time = now
                self.publish_face_result(result)

            return

        # 通过阈值后，不立刻成功，要求连续确认
        if passed:
            if name == self.pass_candidate_name:
                self.pass_confirm_count += 1
            else:
                self.pass_candidate_name = name
                self.pass_confirm_count = 1

            self.get_logger().info(
                f"候选认证：{name} role={role} score={score:.3f} "
                f"confirm={self.pass_confirm_count}/{self.require_pass_frames} "
                f"infer={infer_ms:.1f}ms"
            )

            if self.pass_confirm_count < self.require_pass_frames:
                verifying_result = dict(result)
                verifying_result["passed"] = False
                verifying_result["status"] = "verifying"
                self.publish_face_result(verifying_result)
                return

            self.get_logger().info(
                f"认证通过：{name} role={role} score={score:.3f} infer={infer_ms:.1f}ms"
            )

            result["status"] = "passed"
            result["passed"] = True

            self.publish_face_result(result)

            # 成功后停止本轮认证，防止重复跳转/重复日志
            with self.auth_lock:
                self.auth_active = False

            return

    def publish_face_result(self, result: dict):
        msg = String()

        data = {
            "name": str(result.get("name", "--")),
            "role": str(result.get("role", "--")),
            "score": float(result.get("score", 0.0)),
            "passed": bool(result.get("passed", False)),
            "status": str(result.get("status", "")),
            "infer_ms": float(result.get("infer_ms", 0.0)),
            "result_time": float(result.get("result_time", time.time())),
            "timestamp": now_text(),
        }

        bbox = result.get("bbox", None)
        data["bbox"] = list(bbox) if bbox is not None else None

        error = result.get("error", None)
        if error:
            data["error"] = str(error)

        msg.data = json_dumps(data)
        self.face_result_pub.publish(msg)

    # =====================================================
    # 状态发布
    # =====================================================

    def publish_system_event(self, detail: str, level: str = "info"):
        msg = String()
        msg.data = json_dumps(
            {
                "type": "auth",
                "detail": detail,
                "level": level,
                "timestamp": now_text(),
            }
        )
        self.system_event_pub.publish(msg)

    def publish_camera_status(self):
        msg = String()
        msg.data = json_dumps(
            {
                "device": "face_camera",
                "state": "online" if self.cap is not None else "offline",
                "timestamp": now_text(),
            }
        )
        self.device_status_pub.publish(msg)

    # =====================================================
    # 销毁
    # =====================================================

    def destroy_node(self):
        self.get_logger().info("准备关闭 login_auth_node...")

        self.stop_event.set()
        send_stop_sentinel(self.frame_queue)

        if self.infer_thread is not None:
            self.infer_thread.join(timeout=2.0)
            self.infer_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        print("========== Login Auth ROS2 Node ==========")
        print("权限规则: MJN -> admin，其余已识别人脸 -> user")
        print("onnxruntime:", ort.__version__)
        print("providers:", ort.get_available_providers())

        node = LoginAuthNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出 login_auth_node。")

    except Exception as e:
        print("login_auth_node 启动或运行异常:", repr(e))
        traceback.print_exc()

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("login_auth_node 已退出。")


if __name__ == "__main__":
    main()