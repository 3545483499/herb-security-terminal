#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
herb_recognition_node.py

药草识别 ROS2 节点。

重构要点：
1. 节点随 launch 启动，但默认 standby，不打开摄像头、不执行模型。
2. MainWindow 进入 herb_inventory.ui 时发布 /herb/recognition_control start，本节点才启动工作线程。
3. MainWindow 离开 herb_inventory.ui 时发布 stop，本节点立即置位 stop_event，并由工作线程释放摄像头。
4. 摄像头采集与 ONNX 推理分离到独立线程，ROS 回调不再被 cap.read()/session.run() 阻塞。
5. 采集线程使用 cap.grab() 丢弃旧帧，仅在推流周期到达时 retrieve() 最新帧，降低 V4L2 缓存延迟。
6. ROI 只用于推理裁剪，不在画面上画框；推送给 UI 的画面不叠加文字。
"""

import sys
import site
import json
import time
import threading
import traceback
from pathlib import Path
from collections import deque, Counter

VENV_SITE = Path.home() / "venvs" / "ROS_herb_robot" / "lib" / "python3.12" / "site-packages"
if VENV_SITE.exists():
    site.addsitedir(str(VENV_SITE))
    sys.path.insert(0, str(VENV_SITE))

import cv2
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


DEFAULT_CLASSES = [
    "gancao",
    "gouqi",
    "huangqi",
    "jinyinhua",
    "juhua",
    "kongpan",
    "yiyiren",
]

DEFAULT_LABEL_CN = {
    "gancao": "甘草",
    "gouqi": "枸杞",
    "huangqi": "黄芪",
    "jinyinhua": "金银花",
    "juhua": "菊花",
    "kongpan": "空盘",
    "yiyiren": "薏苡仁",
}

SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


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


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def resize_with_padding(image, target_size=160):
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    result = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return result


def preprocess_bgr(image_bgr, img_size=160, input_type="tensor(float)"):
    image = resize_with_padding(image_bgr, target_size=img_size)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if input_type == "tensor(uint8)":
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image.astype(np.uint8)

    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


def load_classes(classes_path):
    path = Path(classes_path)

    if not path.exists():
        return DEFAULT_CLASSES, DEFAULT_LABEL_CN, 160

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("classes", DEFAULT_CLASSES)
    label_cn = data.get("label_cn", DEFAULT_LABEL_CN)
    img_size = int(data.get("img_size", 160))

    return classes, label_cn, img_size


def parse_roi(roi_text):
    if roi_text is None or str(roi_text).strip() == "":
        return None

    parts = str(roi_text).split(",")
    if len(parts) != 4:
        raise ValueError("ROI 格式错误，应为 x,y,w,h，例如 90,90,350,350")

    x, y, w, h = [int(v.strip()) for v in parts]
    if w <= 0 or h <= 0:
        raise ValueError("ROI 的 w/h 必须大于 0")

    return x, y, w, h


def crop_roi(frame, roi):
    if roi is None:
        return frame

    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]


def create_session(model_path, threads):
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


def decode_result(outputs, classes, label_cn_map, threshold):
    logits = outputs
    if isinstance(logits, list):
        logits = logits[0]
    if logits.ndim == 2:
        logits = logits[0]

    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    if pred_idx >= len(classes):
        label = "unknown"
        label_cn = "未知类别"
    else:
        label = classes[pred_idx]
        label_cn = label_cn_map.get(label, label)

    if conf < threshold:
        result_cn = "低置信度"
        valid = False
    elif label == "kongpan":
        result_cn = "未放置药材"
        valid = False
    else:
        result_cn = label_cn
        valid = True

    return label, result_cn, conf, valid


def stable_vote(history, min_count):
    if len(history) == 0:
        return None

    labels = [item[0] for item in history]
    counter = Counter(labels)
    label, count = counter.most_common(1)[0]

    if count < min_count:
        return None

    same_items = [item for item in history if item[0] == label]
    result_cn = same_items[-1][1]
    valid = same_items[-1][3]
    avg_conf = sum(item[2] for item in same_items) / len(same_items)

    return label, result_cn, avg_conf, valid, count


class HerbRecognitionNode(Node):
    def __init__(self):
        super().__init__("herb_recognition_node")

        self.declare_parameter(
            "model",
            "/home/mjn/herbsecurity_ws/models/herb_type/herb_type_mbv3_160_fp32.onnx",
        )
        self.declare_parameter(
            "classes",
            "/home/mjn/herbsecurity_ws/models/herb_type/classes.json",
        )

        self.declare_parameter("camera", 22)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("camera_fps", 15)
        self.declare_parameter("publish_fps", 15.0)
        self.declare_parameter("infer_interval", 1.0)
        self.declare_parameter("threads", 1)
        self.declare_parameter("threshold", 0.75)
        self.declare_parameter("img_size", 0)
        self.declare_parameter("roi", "90,90,350,350")
        self.declare_parameter("vote_window", 3)
        self.declare_parameter("vote_min_count", 2)
        self.declare_parameter("jpeg_quality", 80)

        self.model_path = Path(str(self.get_parameter("model").value)).expanduser()
        self.classes_path = str(self.get_parameter("classes").value)
        self.camera = int(self.get_parameter("camera").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.publish_fps = float(self.get_parameter("publish_fps").value)
        self.infer_interval = float(self.get_parameter("infer_interval").value)
        self.threads = int(self.get_parameter("threads").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.param_img_size = int(self.get_parameter("img_size").value)
        self.roi_text = str(self.get_parameter("roi").value)
        self.vote_window = int(self.get_parameter("vote_window").value)
        self.vote_min_count = int(self.get_parameter("vote_min_count").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.frame_pub = self.create_publisher(CompressedImage, "/herb/frame/compressed", SENSOR_QOS)
        self.result_pub = self.create_publisher(String, "/herb/recognition_result", 10)
        self.device_status_pub = self.create_publisher(String, "/device/status", 10)
        self.system_event_pub = self.create_publisher(String, "/system/event", 10)
        self.create_subscription(String, "/herb/recognition_control", self.on_control, 10)

        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_type = None
        self.classes = []
        self.label_cn_map = {}
        self.img_size = 160
        self.roi = None

        self.state_lock = threading.RLock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.infer_thread = None
        self.running = False

        self.history = deque(maxlen=self.vote_window)
        self.last_result_key = ""

        self.current_label = "none"
        self.current_result_cn = "--"
        self.current_conf = 0.0
        self.current_infer_ms = 0.0
        self.current_valid = False
        self.current_vote_count = 0

        self.publish_system_event("herb_recognition_node 已启动，等待界面 start 控制", "info")
        self.publish_device_status("standby")

    def on_control(self, msg: String):
        data = json_loads_safe(msg.data)
        cmd = str(data.get("cmd", data.get("raw", ""))).strip().lower()

        if cmd in ["start", "open", "enable", "on"]:
            self.start_recognition()
        elif cmd in ["stop", "close", "disable", "off"]:
            self.stop_recognition()

    def start_recognition(self):
        with self.state_lock:
            if self.running:
                return

            self.reset_runtime_state()
            self.stop_event.clear()
            self.running = True

            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.infer_thread = threading.Thread(target=self.inference_loop, daemon=True)
            self.capture_thread.start()
            self.infer_thread.start()

        self.publish_device_status("starting")
        self.publish_system_event("药草识别线程已启动", "info")
        self.get_logger().info("收到 start，药草识别采集/推理线程已启动。")

    def stop_recognition(self):
        with self.state_lock:
            if not self.running:
                self.reset_runtime_state()
                self.publish_device_status("standby")
                return

            self.running = False
            self.stop_event.set()
            capture_thread = self.capture_thread
            infer_thread = self.infer_thread

        # 不在 ROS 回调里做长时间阻塞；最多等待很短时间。
        for th in [capture_thread, infer_thread]:
            if th is not None and th.is_alive():
                th.join(timeout=0.2)

        self.reset_runtime_state()
        self.publish_device_status("standby")
        self.publish_system_event("药草识别已停止", "info")
        self.get_logger().info("收到 stop，已请求停止药草识别线程。")

    def reset_runtime_state(self):
        with self.frame_lock:
            self.latest_frame = None
            self.latest_frame_seq = 0

        self.history.clear()
        self.last_result_key = ""
        self.current_label = "none"
        self.current_result_cn = "--"
        self.current_conf = 0.0
        self.current_infer_ms = 0.0
        self.current_valid = False
        self.current_vote_count = 0

    def init_model_once(self):
        if self.session is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {self.model_path}")

        self.classes, self.label_cn_map, json_img_size = load_classes(self.classes_path)
        self.img_size = self.param_img_size if self.param_img_size > 0 else json_img_size
        self.roi = parse_roi(self.roi_text)

        self.session = create_session(self.model_path, self.threads)
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        self.input_name = input_info.name
        self.output_name = output_info.name
        self.input_type = input_info.type

        self.get_logger().info(f"中药识别模型: {self.model_path}")
        self.get_logger().info(f"类别: {self.classes}")
        self.get_logger().info(f"输入尺寸: {self.img_size}")
        self.get_logger().info(f"ROI: {self.roi}")
        self.get_logger().info(f"输入: {self.input_name}, {input_info.shape}, {self.input_type}")
        self.get_logger().info(f"输出: {self.output_name}, {output_info.shape}")
        self.get_logger().info(f"providers: {self.session.get_providers()}")

    def open_camera(self):
        cap = cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera)

        if not cap.isOpened():
            raise RuntimeError(f"药草摄像头打开失败: /dev/video{self.camera}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 预热并丢弃旧帧。
        for _ in range(5):
            cap.grab()
            time.sleep(0.01)

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"药草摄像头能打开但读取失败: /dev/video{self.camera}")

        self.get_logger().info(f"药草摄像头打开成功: /dev/video{self.camera}, 首帧={frame.shape}")
        return cap

    def capture_loop(self):
        cap = None
        last_publish_time = 0.0
        min_publish_dt = 1.0 / max(self.publish_fps, 1.0)

        try:
            cap = self.open_camera()
            self.publish_device_status("online")

            while not self.stop_event.is_set():
                grabbed = cap.grab()
                if not grabbed:
                    self.publish_device_status("offline")
                    self.publish_system_event("药草摄像头 grab 失败", "error")
                    time.sleep(0.05)
                    continue

                now = time.time()
                publish_due = now - last_publish_time >= min_publish_dt

                # grab 持续丢帧；只在要推给 UI 时 retrieve 最新物理帧。
                if publish_due:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        self.publish_device_status("offline")
                        self.publish_system_event("药草摄像头 retrieve 失败", "error")
                        time.sleep(0.05)
                        continue

                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.latest_frame_seq += 1

                    self.publish_frame(frame)
                    last_publish_time = now

                time.sleep(0.001)

        except Exception as e:
            self.publish_device_status("offline")
            self.publish_system_event(f"药草采集线程异常: {repr(e)}", "error")
            self.get_logger().error(f"药草采集线程异常: {repr(e)}")
            traceback.print_exc()

        finally:
            if cap is not None:
                cap.release()
            self.get_logger().info("药草采集线程已退出，摄像头已释放。")

    def inference_loop(self):
        last_infer_time = 0.0
        last_seq = -1

        try:
            self.init_model_once()

            while not self.stop_event.is_set():
                now = time.time()
                if now - last_infer_time < self.infer_interval:
                    time.sleep(0.01)
                    continue

                with self.frame_lock:
                    frame = None if self.latest_frame is None else self.latest_frame.copy()
                    seq = self.latest_frame_seq

                if frame is None or seq == last_seq:
                    time.sleep(0.02)
                    continue

                last_seq = seq
                last_infer_time = now
                self.run_inference(frame)

        except Exception as e:
            self.publish_device_status("offline")
            self.publish_system_event(f"药草推理线程异常: {repr(e)}", "error")
            self.get_logger().error(f"药草推理线程异常: {repr(e)}")
            traceback.print_exc()

        finally:
            self.get_logger().info("药草推理线程已退出。")

    def run_inference(self, frame):
        infer_img = crop_roi(frame, self.roi)
        input_tensor = preprocess_bgr(
            infer_img,
            img_size=self.img_size,
            input_type=self.input_type,
        )

        if input_tensor is None:
            return

        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        t1 = time.perf_counter()
        self.current_infer_ms = (t1 - t0) * 1000.0

        label, result_cn, conf, valid = decode_result(
            outputs=outputs,
            classes=self.classes,
            label_cn_map=self.label_cn_map,
            threshold=self.threshold,
        )

        self.history.append((label, result_cn, conf, valid))
        stable = stable_vote(self.history, self.vote_min_count)

        if stable is not None:
            stable_label, stable_result_cn, stable_conf, stable_valid, vote_count = stable
            self.current_label = stable_label
            self.current_result_cn = stable_result_cn
            self.current_conf = stable_conf
            self.current_valid = stable_valid
            self.current_vote_count = vote_count
        else:
            self.current_label = label
            self.current_result_cn = result_cn
            self.current_conf = conf
            self.current_valid = valid
            self.current_vote_count = 1

        self.publish_result_if_changed()

    def publish_frame(self, frame):
        # 重要：发布原始画面，不画 ROI 框，不写识别字符。
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "herb_camera"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.frame_pub.publish(msg)

    def publish_result_if_changed(self):
        key = f"{self.current_label}:{self.current_result_cn}:{self.current_valid}:{self.current_conf:.3f}:{self.current_vote_count}"
        if key == self.last_result_key:
            return

        self.last_result_key = key
        self.publish_result()

    def publish_result(self):
        msg = String()
        msg.data = json_dumps(
            {
                "herb_name": self.current_result_cn,
                "name": self.current_result_cn,
                "class_name": self.current_label,
                "label": self.current_label,
                "label_cn": self.current_result_cn,
                "confidence": round(float(self.current_conf), 4),
                "score": round(float(self.current_conf), 4),
                "valid": bool(self.current_valid),
                "status": "valid" if self.current_valid else "invalid",
                "infer_ms": round(float(self.current_infer_ms), 2),
                "vote_count": int(self.current_vote_count),
                "vote_window": int(len(self.history)),
                "roi": self.roi_text,
                "sensor": "herb_camera",
                "timestamp": now_text(),
            }
        )
        self.result_pub.publish(msg)

    def publish_device_status(self, state: str):
        msg = String()
        msg.data = json_dumps(
            {
                "device": "herb_camera",
                "state": state,
                "camera": self.camera,
                "timestamp": now_text(),
            }
        )
        self.device_status_pub.publish(msg)

    def publish_system_event(self, detail: str, level: str = "info"):
        msg = String()
        msg.data = json_dumps(
            {
                "type": "herb_recognition",
                "detail": detail,
                "level": level,
                "timestamp": now_text(),
            }
        )
        self.system_event_pub.publish(msg)

    def destroy_node(self):
        self.stop_recognition()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        print("========== Herb Recognition ROS2 Node ==========")
        print("onnxruntime:", ort.__version__)
        print("providers:", ort.get_available_providers())
        node = HerbRecognitionNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出 herb_recognition_node。")

    except Exception as e:
        print("herb_recognition_node 异常:", repr(e))
        traceback.print_exc()

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("herb_recognition_node 已退出。")


if __name__ == "__main__":
    main()
