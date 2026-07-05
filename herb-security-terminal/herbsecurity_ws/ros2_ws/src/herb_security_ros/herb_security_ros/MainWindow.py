#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MainWindow.py

中药材称重识别与环境控制终端。

本版关键点：
1. herb_recognition_node 默认不运行模型；只有进入 herb_inventory.ui 才发布 start。
2. 离开 herb_inventory.ui 时发布 stop，释放药草摄像头并停止推理。
3. GUI 进程不 import cv2，避免 PyQt5 与 OpenCV Qt 插件冲突。
4. 出入库页摄像头画面不显示 OpenCV 叠加文字、不显示 ROI 框；这些由 herb_recognition_node 保证。
5. 出入库页顶部说明文字隐藏，减少界面杂乱。
6. 保留日志 O(1)、Dirty Check、FastTransformation、ROS 30ms spin 优化。
"""

import sys
import os
import re
import json
import time
import signal
import tempfile
from pathlib import Path
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from PyQt5 import uic
from PyQt5.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
    QMetaObject,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QTextCursor,
    QPainter,
    QPainterPath,
    QBrush,
    QColor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
)

from ament_index_python.packages import get_package_share_directory


PACKAGE_NAME = "herb_security_ros"

MAIN_UI_NAME = "mainwindow.ui"
LOGIN_UI_NAME = "login_auth.ui"
HERB_INVENTORY_UI_NAME = "herb_inventory.ui"

DEFAULT_HERBS = ["甘草", "枸杞", "黄芪", "金银花", "菊花", "薏苡仁"]

SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in [
            "true",
            "1",
            "yes",
            "on",
            "passed",
            "pass",
            "success",
            "ok",
            "normal",
            "valid",
            "正常",
            "通过",
            "稳定",
        ]

    if isinstance(value, (int, float)):
        return value != 0

    return default


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


def role_to_cn(role: str) -> str:
    role = str(role).strip()
    if role in ["admin", "administrator", "root", "管理员"]:
        return "管理员"
    if role in ["user", "normal", "普通", "普通用户"]:
        return "普通用户"
    return role if role else "--"


def is_admin_role(role: str) -> bool:
    return str(role).strip() in ["admin", "administrator", "root", "管理员"]


def bool_property(obj, name: str) -> bool:
    return bool(obj.property(name))


def make_ui_compatible_copy(ui_path: Path) -> str:
    text = ui_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r'<sizepolicy\s+hsizetype="([^"]+)"\s+vsizetype="([^"]+)"\s*/>'
    )

    fixed = pattern.sub(
        r'<sizepolicy hsizetype="\1" vsizetype="\2">\n'
        r' <horstretch>0</horstretch>\n'
        r' <verstretch>0</verstretch>\n'
        r'</sizepolicy>',
        text,
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=ui_path.stem + "_compatible_",
        suffix=".ui",
        text=True,
    )

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(fixed)

    return temp_path


def load_ui_compatible(ui_path: Path, baseinstance=None):
    temp_path = make_ui_compatible_copy(ui_path)
    try:
        if baseinstance is None:
            return uic.loadUi(temp_path)
        return uic.loadUi(temp_path, baseinstance)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def get_ui_path(ui_name: str) -> Path:
    package_share = Path(get_package_share_directory(PACKAGE_NAME))
    install_path = package_share / "ui" / ui_name
    if install_path.exists():
        return install_path

    this_file = Path(__file__).resolve()
    src_pkg_root = this_file.parent.parent
    src_path = src_pkg_root / "ui" / ui_name
    if src_path.exists():
        return src_path

    raise FileNotFoundError(
        f"找不到 UI 文件：{ui_name}\n"
        f"已尝试：\n  {install_path}\n  {src_path}"
    )


class RosWorker(QObject):
    sig_auth_result = pyqtSignal(str)
    sig_herb_result = pyqtSignal(str)
    sig_weight = pyqtSignal(str)
    sig_env_status = pyqtSignal(str)
    sig_inventory_status = pyqtSignal(str)
    sig_door_state = pyqtSignal(str)
    sig_device_status = pyqtSignal(str)
    sig_system_event = pyqtSignal(str)

    sig_auth_frame = pyqtSignal(QImage)
    sig_herb_frame = pyqtSignal(QImage)

    sig_log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.node = None
        self.spin_timer = None
        self.stopping = False

        self.auth_request_pub = None
        self.auth_logout_pub = None
        self.herb_control_pub = None
        self.inventory_action_pub = None
        self.door_cmd_pub = None
        self.device_cmd_pub = None
        self.ui_event_pub = None

        self.max_ui_fps = 15
        self.last_auth_frame_time = 0.0
        self.last_herb_frame_time = 0.0

    @pyqtSlot()
    def start(self):
        try:
            self.node = rclpy.create_node("main_window_ui_node")

            self.auth_request_pub = self.node.create_publisher(String, "/auth/request", 10)
            self.auth_logout_pub = self.node.create_publisher(String, "/auth/logout", 10)
            self.herb_control_pub = self.node.create_publisher(String, "/herb/recognition_control", 10)
            self.inventory_action_pub = self.node.create_publisher(String, "/inventory/action", 10)
            self.door_cmd_pub = self.node.create_publisher(String, "/control/door_cmd", 10)
            self.device_cmd_pub = self.node.create_publisher(String, "/control/device_cmd", 10)
            self.ui_event_pub = self.node.create_publisher(String, "/system/ui_event", 10)

            self.node.create_subscription(String, "/auth/face_result", self._on_auth_result, 10)
            self.node.create_subscription(String, "/herb/recognition_result", self._on_herb_result, 10)
            self.node.create_subscription(String, "/weight/current", self._on_weight, 10)
            self.node.create_subscription(String, "/env/status", self._on_env_status, 10)
            self.node.create_subscription(String, "/inventory/status", self._on_inventory_status, 10)
            self.node.create_subscription(String, "/control/door_state", self._on_door_state, 10)
            self.node.create_subscription(String, "/device/status", self._on_device_status, 10)
            self.node.create_subscription(String, "/system/event", self._on_system_event, 10)

            self.node.create_subscription(
                CompressedImage,
                "/auth/face_frame/compressed",
                self._on_auth_frame,
                SENSOR_QOS,
            )
            self.node.create_subscription(
                CompressedImage,
                "/herb/frame/compressed",
                self._on_herb_frame,
                SENSOR_QOS,
            )

            self.spin_timer = QTimer(self)
            self.spin_timer.setInterval(30)
            self.spin_timer.timeout.connect(self.spin_once)
            self.spin_timer.start()

            self.sig_log.emit("主界面 ROS2 节点已启动")

        except Exception as e:
            self.sig_log.emit(f"主界面 ROS2 节点启动失败：{e}")
            self.finished.emit()

    @pyqtSlot()
    def spin_once(self):
        if self.stopping or self.node is None:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        except Exception as e:
            self.sig_log.emit(f"ROS spin 异常：{e}")

    @pyqtSlot()
    def stop(self):
        if self.stopping:
            return
        self.stopping = True

        self.publish_herb_control("stop")
        self.publish_ui_event({"type": "shutdown", "detail": "主界面请求安全退出", "timestamp": now_text()})
        self.publish_device_cmd({"device": "all", "cmd": "safe_stop", "reason": "ui_shutdown", "timestamp": now_text()})

        if self.spin_timer is not None:
            try:
                self.spin_timer.stop()
            except Exception:
                pass
            self.spin_timer = None

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception as e:
                self.sig_log.emit(f"ROS 节点销毁异常：{e}")
            self.node = None

        self.sig_log.emit("主界面 ROS2 节点已停止")
        self.finished.emit()

    def _publish_json(self, publisher, data: dict):
        if publisher is None:
            return
        msg = String()
        msg.data = json_dumps(data)
        publisher.publish(msg)

    @pyqtSlot()
    def publish_auth_request(self):
        self._publish_json(self.auth_request_pub, {"cmd": "start", "source": "mainwindow", "timestamp": now_text()})

    @pyqtSlot()
    def publish_auth_logout(self):
        self._publish_json(self.auth_logout_pub, {"cmd": "logout", "source": "mainwindow", "timestamp": now_text()})

    @pyqtSlot(str)
    def publish_herb_control_slot(self, cmd: str):
        self.publish_herb_control(cmd)

    def publish_herb_control(self, cmd: str):
        self._publish_json(
            self.herb_control_pub,
            {
                "cmd": cmd,
                "source": "mainwindow",
                "camera": 22,
                "roi": "90,90,350,350",
                "timestamp": now_text(),
            },
        )

    @pyqtSlot(str, str)
    def publish_inventory_action(self, action: str, payload_text: str):
        payload = json_loads_safe(payload_text)
        payload["action"] = action
        payload["source"] = "mainwindow"
        payload["timestamp"] = now_text()
        self._publish_json(self.inventory_action_pub, payload)

    @pyqtSlot(str, str, str)
    def publish_door_cmd(self, cmd: str, reason: str, operator: str):
        self._publish_json(
            self.door_cmd_pub,
            {"cmd": cmd, "reason": reason, "operator": operator, "source": "mainwindow", "timestamp": now_text()},
        )

    @pyqtSlot(str, str, str, str)
    def publish_device_cmd_slot(self, device: str, cmd: str, reason: str, operator: str):
        self.publish_device_cmd(
            {"device": device, "cmd": cmd, "reason": reason, "operator": operator, "source": "mainwindow", "timestamp": now_text()}
        )

    def publish_device_cmd(self, data: dict):
        self._publish_json(self.device_cmd_pub, data)

    @pyqtSlot(str, str)
    def publish_ui_event_slot(self, event_type: str, detail: str):
        self.publish_ui_event({"type": event_type, "detail": detail, "timestamp": now_text()})

    def publish_ui_event(self, data: dict):
        self._publish_json(self.ui_event_pub, data)

    def _on_auth_result(self, msg: String):
        self.sig_auth_result.emit(msg.data)

    def _on_herb_result(self, msg: String):
        self.sig_herb_result.emit(msg.data)

    def _on_weight(self, msg: String):
        self.sig_weight.emit(msg.data)

    def _on_env_status(self, msg: String):
        self.sig_env_status.emit(msg.data)

    def _on_inventory_status(self, msg: String):
        self.sig_inventory_status.emit(msg.data)

    def _on_door_state(self, msg: String):
        self.sig_door_state.emit(msg.data)

    def _on_device_status(self, msg: String):
        self.sig_device_status.emit(msg.data)

    def _on_system_event(self, msg: String):
        self.sig_system_event.emit(msg.data)

    def _on_auth_frame(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self.last_auth_frame_time < 1.0 / 10:
            return
        self.last_auth_frame_time = now
        qimg = self.decode_compressed_image(msg.data)
        if not qimg.isNull():
            self.sig_auth_frame.emit(qimg)

    def _on_herb_frame(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self.last_herb_frame_time < 1.0 / max(self.max_ui_fps, 1):
            return
        self.last_herb_frame_time = now
        qimg = self.decode_compressed_image(msg.data)
        if not qimg.isNull():
            self.sig_herb_frame.emit(qimg)

    def decode_compressed_image(self, data) -> QImage:
        try:
            raw = bytes(data)
            image = QImage.fromData(raw, "JPG")
            if image.isNull():
                image = QImage.fromData(raw, "JPEG")
            if image.isNull():
                image = QImage.fromData(raw, "PNG")
            if image.isNull():
                image = QImage.fromData(raw)
            if image.isNull():
                return QImage()
            return image.convertToFormat(QImage.Format_RGB888).copy()
        except Exception as e:
            self.sig_log.emit(f"图像解码失败：{e}")
            return QImage()


class LoginAuthPageController(QObject):
    sig_back = pyqtSignal()
    sig_password_login = pyqtSignal()
    sig_login_success = pyqtSignal(dict)
    sig_login_failed = pyqtSignal(dict)

    def __init__(self, page: QWidget):
        super().__init__()
        self.page = page
        self.last_frame = QImage()
        self.bind_buttons()
        self.set_idle()

    def label(self, name: str):
        return self.page.findChild(QLabel, name)

    def button(self, name: str):
        return self.page.findChild(QPushButton, name)

    def set_label(self, name: str, text: str):
        obj = self.label(name)
        if obj is not None and obj.text() != str(text):
            obj.setText(str(text))

    def bind_buttons(self):
        btn_back = self.button("btn_back")
        if btn_back is not None:
            btn_back.clicked.connect(self.sig_back.emit)
        btn_password = self.button("btn_password_login")
        if btn_password is not None:
            btn_password.clicked.connect(self.sig_password_login.emit)

    def set_auth_state(self, text: str, state: str):
        label = self.label("label_auth_result_value")
        if label is None:
            return
        text = str(text)
        state = str(state)
        text_changed = label.text() != text
        state_changed = label.property("state") != state
        if not text_changed and not state_changed:
            return
        if text_changed:
            label.setText(text)
        if state_changed:
            label.setProperty("state", state)
            label.style().unpolish(label)
            label.style().polish(label)
        label.update()

    def set_idle(self):
        self.set_label("label_auth_name_value", "姓名：--")
        self.set_label("label_auth_role_value", "权限：--")
        self.set_label("label_auth_guide_value", "请注视摄像头，保持面部在画面中央。")
        self.set_auth_state("未认证", "idle")
        avatar = self.label("label_auth_avatar")
        if avatar is not None:
            avatar.clear()
            avatar.setText("头像")
        camera = self.label("label_camera_view")
        if camera is not None:
            camera.clear()
            camera.setText("CAMERA PREVIEW")

    def set_running(self):
        self.set_label("label_auth_guide_value", "正在识别人脸，请保持面部在画面中央。")
        self.set_auth_state("识别中", "running")

    @pyqtSlot(str)
    def on_auth_result(self, text: str):
        data = json_loads_safe(text)
        name = str(data.get("name", "--"))
        role = str(data.get("role", "--"))
        passed = safe_bool(data.get("passed", False))
        score = safe_float(data.get("score", 0.0))
        status = str(data.get("status", ""))

        if name in ["NoFace", "AlignFail"] or status in ["noface", "align_fail"]:
            self.set_label("label_auth_name_value", "姓名：--")
            self.set_label("label_auth_role_value", "权限：--")
            self.set_label("label_auth_guide_value", "未检测到有效人脸，请靠近摄像头并保持正对。")
            self.set_auth_state("等待人脸", "running")
            return

        if status == "verifying":
            self.set_label("label_auth_name_value", f"姓名：{name}")
            self.set_label("label_auth_role_value", f"权限：{role_to_cn(role)}")
            self.set_label("label_auth_guide_value", f"识别到 {name}，正在连续确认，score={score:.3f}")
            self.set_auth_state("确认中", "running")
            return

        if passed and status == "passed":
            self.set_label("label_auth_name_value", f"姓名：{name}")
            self.set_label("label_auth_role_value", f"权限：{role_to_cn(role)}")
            self.set_label("label_auth_guide_value", f"识别分数：{score:.3f}，认证通过。")
            self.set_auth_state("认证通过", "success")
            self.update_avatar_from_last_frame()
            self.sig_login_success.emit({"name": name, "role": role, "passed": True, "score": score})
            return

        self.set_label("label_auth_name_value", "姓名：--")
        self.set_label("label_auth_role_value", "权限：--")
        self.set_label("label_auth_guide_value", f"认证失败，score={score:.3f}，请重新认证。")
        self.set_auth_state("认证失败", "fail")
        self.sig_login_failed.emit({"name": name, "role": role, "passed": False, "score": score})

    @pyqtSlot(QImage)
    def on_auth_frame(self, image: QImage):
        if image.isNull():
            return
        self.last_frame = image.copy()
        label = self.label("label_camera_view")
        if label is not None:
            render_image_to_label(label, image)

    def update_avatar_from_last_frame(self):
        if self.last_frame.isNull():
            return
        avatar = self.label("label_auth_avatar")
        if avatar is None:
            return
        size = min(avatar.width(), avatar.height())
        if size <= 0:
            size = 72
        pixmap = make_circle_pixmap(self.last_frame, size)
        avatar.setText("")
        avatar.setPixmap(pixmap)


def render_image_to_label(label: QLabel, image: QImage):
    if label is None or image.isNull():
        return
    label_w = label.width()
    label_h = label.height()
    if label_w <= 0 or label_h <= 0:
        return
    pixmap = QPixmap.fromImage(image)
    scaled = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
    x = max(0, (scaled.width() - label_w) // 2)
    y = max(0, (scaled.height() - label_h) // 2)
    cropped = scaled.copy(x, y, label_w, label_h)
    label.setPixmap(cropped)


def make_circle_pixmap(image: QImage, size: int) -> QPixmap:
    if image.isNull():
        return QPixmap()
    source = QPixmap.fromImage(image)
    scaled = source.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
    x = max(0, (scaled.width() - size) // 2)
    y = max(0, (scaled.height() - size) // 2)
    square = scaled.copy(x, y, size, size)
    result = QPixmap(size, size)
    result.fill(Qt.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, square)
    painter.end()
    return result


class HerbInventoryPageController(QObject):
    sig_back = pyqtSignal()
    sig_stock_in = pyqtSignal()
    sig_stock_out = pyqtSignal()

    def __init__(self, page: QWidget):
        super().__init__()
        self.page = page
        self.last_frame = QImage()
        self.bind_buttons()
        self.init_table()
        self.hide_extra_text()
        self.set_idle()

    def label(self, name: str):
        return self.page.findChild(QLabel, name)

    def button(self, name: str):
        return self.page.findChild(QPushButton, name)

    def table(self, name: str):
        return self.page.findChild(QTableWidget, name)

    def set_label(self, name: str, text: str):
        obj = self.label(name)
        if obj is not None and obj.text() != str(text):
            obj.setText(str(text))

    def hide_extra_text(self):
        # 删除/隐藏出入库管理页上方说明字符，保留返回按钮。
        for name in ["label_page_subtitle", "label_camera_title"]:
            obj = self.label(name)
            if obj is not None:
                obj.hide()

        title = self.label("label_page_title")
        if title is not None:
            title.setText("药草出入库")

    def bind_buttons(self):
        btn_back = self.button("btn_back")
        if btn_back is not None:
            btn_back.clicked.connect(self.sig_back.emit)
        btn_stock_in = self.button("btn_stock_in")
        if btn_stock_in is not None:
            btn_stock_in.clicked.connect(self.sig_stock_in.emit)
        btn_stock_out = self.button("btn_stock_out")
        if btn_stock_out is not None:
            btn_stock_out.clicked.connect(self.sig_stock_out.emit)

    def init_table(self):
        table = self.table("table_inventory")
        if table is None:
            return
        table.setRowCount(len(DEFAULT_HERBS))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["药草", "库存/g", "状态"])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        try:
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
        except Exception:
            pass

    def set_idle(self):
        self.set_label("label_herb_type_value", "--")
        self.set_label("label_confidence_value", "--")
        self.set_label("label_weight_value", "0.0 g")
        self.set_stable(False)
        self.set_buttons_enabled(False)
        camera = self.label("label_camera_view")
        if camera is not None:
            camera.clear()
            camera.setText("")

    def reset_recognition(self, clear_camera: bool = True):
        self.set_label("label_herb_type_value", "--")
        self.set_label("label_confidence_value", "--")
        self.set_buttons_enabled(False)

        if clear_camera:
            camera = self.label("label_camera_view")
            if camera is not None:
                camera.clear()
                camera.setText("")

    def set_stable(self, stable: bool):
        label = self.label("label_stable_value")
        if label is None:
            return
        text = "稳定" if stable else "未稳定"
        state = "stable" if stable else "unstable"
        text_changed = label.text() != text
        state_changed = label.property("state") != state
        if not text_changed and not state_changed:
            return
        if text_changed:
            label.setText(text)
        if state_changed:
            label.setProperty("state", state)
            label.style().unpolish(label)
            label.style().polish(label)
        label.update()

    def set_buttons_enabled(self, enabled: bool):
        for name in ["btn_stock_in", "btn_stock_out"]:
            btn = self.button(name)
            if btn is not None and btn.isEnabled() != bool(enabled):
                btn.setEnabled(bool(enabled))

    def update_recognition(self, herb_name: str, confidence: float, valid: bool):
        if valid:
            self.set_label("label_herb_type_value", herb_name)
            self.set_label("label_confidence_value", f"{confidence:.3f}")
        else:
            self.set_label("label_herb_type_value", "--")
            self.set_label("label_confidence_value", "--")

    def update_weight(self, weight: float, stable: bool, unit: str = "g"):
        self.set_label("label_weight_value", f"{weight:.1f} {unit}")
        self.set_stable(stable)

    @pyqtSlot(QImage)
    def on_frame(self, image: QImage):
        if image.isNull():
            return
        self.last_frame = image.copy()
        label = self.label("label_camera_view")
        if label is not None:
            render_image_to_label(label, image)

    def refresh_inventory_table(self, inventory: dict, current_herb: str = "--"):
        table = self.table("table_inventory")
        if table is None:
            return

        # 固定行顺序，避免识别波动导致表格行跳变和误触。
        herbs = list(DEFAULT_HERBS)

        highlight_brush = QBrush(QColor(29, 78, 58))
        normal_brush = QBrush()

        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(herbs))
            for row, herb in enumerate(herbs):
                info = inventory.get(herb, {"weight": 0.0, "threshold": 50.0, "status": "正常"})
                weight = safe_float(info.get("weight", 0.0))
                threshold = safe_float(info.get("threshold", 50.0))
                status = "低库存" if weight < threshold else "正常"
                values = [herb, f"{weight:.1f}", status]
                is_current = current_herb == herb

                for col, value in enumerate(values):
                    item = table.item(row, col)
                    if item is None:
                        item = QTableWidgetItem(value)
                        item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(row, col, item)
                    elif item.text() != value:
                        item.setText(value)

                    item.setBackground(highlight_brush if is_current else normal_brush)
        finally:
            table.setUpdatesEnabled(True)


class MainWindow(QMainWindow):
    sig_request_ros_stop = pyqtSignal()
    sig_auth_request = pyqtSignal()
    sig_auth_logout = pyqtSignal()
    sig_herb_control = pyqtSignal(str)
    sig_inventory_action = pyqtSignal(str, str)
    sig_door_cmd = pyqtSignal(str, str, str)
    sig_device_cmd = pyqtSignal(str, str, str, str)
    sig_ui_event = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.shutdown_requested = False
        self.allow_close = False
        self.max_log_count = 80
        self.last_fail_log_time = 0.0
        self.herb_page_active = False
        self.current_user = {"name": "--", "role": "--", "passed": False, "score": 0.0}
        self.current_herb = {"herb_name": "--", "confidence": 0.0, "valid": False}
        self.current_weight = {"weight": 0.0, "stable": False, "unit": "g"}
        self.inventory = {herb: {"weight": 0.0, "threshold": 50.0, "status": "正常"} for herb in DEFAULT_HERBS}
        self.load_pages()
        self.init_window()
        self.init_tables()
        self.bind_buttons()
        self.init_timers()
        self.init_default_state()

    def load_pages(self):
        main_ui_path = get_ui_path(MAIN_UI_NAME)
        login_ui_path = get_ui_path(LOGIN_UI_NAME)
        herb_inventory_ui_path = get_ui_path(HERB_INVENTORY_UI_NAME)
        load_ui_compatible(main_ui_path, self)
        self.main_page = self.takeCentralWidget()
        self.login_shell = load_ui_compatible(login_ui_path)
        self.login_page = self.login_shell.takeCentralWidget()
        self.login_page.setStyleSheet(self.login_shell.styleSheet())
        self.herb_inventory_page = load_ui_compatible(herb_inventory_ui_path)
        self.login_auth = LoginAuthPageController(self.login_page)
        self.herb_inventory_ui = HerbInventoryPageController(self.herb_inventory_page)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.herb_inventory_page)
        self.setCentralWidget(self.stack)
        self.stack.setCurrentWidget(self.main_page)

    def main_label(self, name: str):
        return self.main_page.findChild(QLabel, name)

    def main_button(self, name: str):
        return self.main_page.findChild(QPushButton, name)

    def main_plain(self, name: str):
        return self.main_page.findChild(QPlainTextEdit, name)

    def main_table(self, name: str):
        return self.main_page.findChild(QTableWidget, name)

    def main_stack(self):
        return self.main_page.findChild(QStackedWidget, "stacked_main_pages")

    def main_widget(self, name: str):
        return self.main_page.findChild(QWidget, name)

    def main_check(self, name: str):
        return self.main_page.findChild(QCheckBox, name)

    def set_main_label(self, name: str, text: str):
        label = self.main_label(name)
        if label is not None and label.text() != str(text):
            label.setText(str(text))

    def set_chip(self, name: str, text: str, alarm: bool = False, warn: bool = False):
        label = self.main_label(name)
        if label is None:
            return
        text = str(text)
        alarm = bool(alarm)
        warn = bool(warn)
        text_changed = label.text() != text
        alarm_changed = bool_property(label, "alarm") != alarm
        warn_changed = bool_property(label, "warn") != warn
        if not text_changed and not alarm_changed and not warn_changed:
            return
        if text_changed:
            label.setText(text)
        if alarm_changed or warn_changed:
            label.setProperty("alarm", alarm)
            label.setProperty("warn", warn)
            label.style().unpolish(label)
            label.style().polish(label)
        label.update()

    def set_realtime_status(self, text: str, alarm: bool = False):
        label = self.main_label("label_realtime_status_value")
        if label is None:
            return
        text = str(text)
        alarm = bool(alarm)
        text_changed = label.text() != text
        alarm_changed = bool_property(label, "alarm") != alarm
        if not text_changed and not alarm_changed:
            return
        if text_changed:
            label.setText(text)
        if alarm_changed:
            label.setProperty("alarm", alarm)
            label.style().unpolish(label)
            label.style().polish(label)
        label.update()

    def set_button_enabled(self, name: str, enabled: bool):
        btn = self.main_button(name)
        if btn is not None and btn.isEnabled() != bool(enabled):
            btn.setEnabled(bool(enabled))

    def init_window(self):
        self.setWindowTitle("中药材称重识别与环境控制终端")

    def init_tables(self):
        table = self.main_table("table_inventory")
        if table is not None:
            table.setRowCount(len(DEFAULT_HERBS))
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["药草", "库存/g", "阈值/g", "状态"])
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            try:
                table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                table.verticalHeader().setVisible(False)
            except Exception:
                pass
        log_table = self.main_table("table_system_logs")
        if log_table is not None:
            log_table.setColumnCount(4)
            log_table.setHorizontalHeaderLabels(["时间", "类型", "人员", "内容"])
            log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            log_table.setSelectionMode(QAbstractItemView.SingleSelection)
            try:
                log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                log_table.verticalHeader().setVisible(False)
            except Exception:
                pass
        plain = self.main_plain("plain_local_log")
        if plain is not None:
            plain.setMaximumBlockCount(self.max_log_count)
            plain.setReadOnly(True)

    def bind_buttons(self):
        btn_login = self.main_button("btn_lock_toggle")
        if btn_login is not None:
            btn_login.clicked.connect(self.on_open_login_page)
        btn_start_auth = self.main_button("btn_start_auth")
        if btn_start_auth is not None:
            btn_start_auth.clicked.connect(self.on_open_login_page)
        btn_logout_auth = self.main_button("btn_logout_auth")
        if btn_logout_auth is not None:
            btn_logout_auth.clicked.connect(self.on_logout_auth)
        btn_exit = self.main_button("btn_exit_system")
        if btn_exit is not None:
            btn_exit.clicked.connect(self.on_exit_system)
        self.login_auth.sig_back.connect(self.on_back_to_main)
        self.login_auth.sig_password_login.connect(self.on_password_login_demo)
        self.login_auth.sig_login_success.connect(self.on_login_success)
        self.login_auth.sig_login_failed.connect(self.on_login_failed)
        self.herb_inventory_ui.sig_back.connect(self.on_back_to_main)
        self.herb_inventory_ui.sig_stock_in.connect(lambda: self.on_inventory_action("stock_in"))
        self.herb_inventory_ui.sig_stock_out.connect(lambda: self.on_inventory_action("stock_out"))
        self.bind_inner_page_button("btn_home_overview", "page_home")
        btn_herb_inventory = self.main_button("btn_herb_inventory")
        if btn_herb_inventory is not None:
            btn_herb_inventory.clicked.connect(self.on_open_herb_inventory_page)
        self.bind_inner_page_button("btn_env_control", "page_env_control")
        self.bind_inner_page_button("btn_log_trace", "page_logs")
        self.bind_inner_page_button("btn_admin_settings", "page_admin")
        btn_stock_in = self.main_button("btn_stock_in")
        if btn_stock_in is not None:
            btn_stock_in.clicked.connect(lambda: self.on_inventory_action("stock_in"))
        btn_stock_out = self.main_button("btn_stock_out")
        if btn_stock_out is not None:
            btn_stock_out.clicked.connect(lambda: self.on_inventory_action("stock_out"))
        for name, cmd in [("btn_gate_open", "open"), ("btn_gate_close", "close")]:
            btn = self.main_button(name)
            if btn is not None:
                btn.clicked.connect(lambda checked=False, c=cmd: self.on_gate_cmd(c))
        for name, device, cmd in [
            ("btn_fan_start", "fan", "on"),
            ("btn_fan_stop", "fan", "off"),
            ("btn_alarm_start", "alarm", "on"),
            ("btn_alarm_stop", "alarm", "off"),
        ]:
            btn = self.main_button(name)
            if btn is not None:
                btn.clicked.connect(lambda checked=False, d=device, c=cmd: self.on_device_cmd(d, c, "manual"))
        for name in ["btn_admin_add_user", "btn_admin_role", "btn_admin_inventory", "btn_admin_threshold"]:
            btn = self.main_button(name)
            if btn is not None:
                btn.clicked.connect(lambda checked=False, n=name: self.on_admin_action(n))
        check_gate = self.main_check("check_gate_override")
        if check_gate is not None:
            check_gate.stateChanged.connect(lambda _: self.update_permission_buttons())

    def bind_inner_page_button(self, button_name: str, page_name: str):
        btn = self.main_button(button_name)
        if btn is None:
            return
        btn.clicked.connect(lambda: self.switch_inner_page(page_name))

    def switch_inner_page(self, page_name: str):
        self.stop_herb_page_if_needed()
        stack = self.main_stack()
        page = self.main_widget(page_name)
        if stack is not None and page is not None:
            stack.setCurrentWidget(page)
        self.sig_ui_event.emit("main_page", page_name)

    def init_timers(self):
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_time)
        self.clock_timer.start(1000)
        self.update_time()

    def init_default_state(self):
        self.set_chip("label_wifi_status", "WIFI 在线")
        self.set_chip("label_facecam_status", "人脸 待机", warn=True)
        self.set_chip("label_herbcam_status", "药草 待机", warn=True)
        self.set_chip("label_aht20_status", "AHT30 待机", warn=True)
        self.set_chip("label_hx711_status", "HX711 待机", warn=True)
        self.set_chip("label_gate_servo_status", "舵机 关闭", warn=True)
        self.set_chip("label_fan_status", "风扇 待命", warn=True)
        self.set_chip("label_alarm_status", "声光 待命", warn=True)
        self.set_realtime_status("系统锁定")
        self.update_auth_display()
        self.update_herb_display()
        self.update_weight_display()
        self.refresh_inventory_table()
        self.update_permission_buttons()
        self.add_log("系统启动完成，等待登录认证。", "系统")

    def connect_ros_worker(self, worker: RosWorker):
        self.sig_request_ros_stop.connect(worker.stop)
        self.sig_auth_request.connect(worker.publish_auth_request)
        self.sig_auth_logout.connect(worker.publish_auth_logout)
        self.sig_herb_control.connect(worker.publish_herb_control_slot)
        self.sig_inventory_action.connect(worker.publish_inventory_action)
        self.sig_door_cmd.connect(worker.publish_door_cmd)
        self.sig_device_cmd.connect(worker.publish_device_cmd_slot)
        self.sig_ui_event.connect(worker.publish_ui_event_slot)
        worker.sig_auth_result.connect(self.login_auth.on_auth_result)
        worker.sig_auth_frame.connect(self.login_auth.on_auth_frame)
        worker.sig_herb_frame.connect(self.on_herb_frame)
        worker.sig_herb_result.connect(self.on_herb_result)
        worker.sig_weight.connect(self.on_weight)
        worker.sig_env_status.connect(self.on_env_status)
        worker.sig_inventory_status.connect(self.on_inventory_status)
        worker.sig_door_state.connect(self.on_door_state)
        worker.sig_device_status.connect(self.on_device_status)
        worker.sig_system_event.connect(self.on_system_event)
        worker.sig_log.connect(lambda text: self.add_log(text, "ROS"))

    def reset_herb_runtime_state(self):
        self.current_herb = {
            "herb_name": "--",
            "confidence": 0.0,
            "valid": False,
        }
        self.update_herb_display()
        self.herb_inventory_ui.reset_recognition(clear_camera=True)
        self.herb_inventory_ui.refresh_inventory_table(self.inventory, "--")
        self.update_permission_buttons()

    def stop_herb_page_if_needed(self):
        if self.herb_page_active:
            self.herb_page_active = False
            self.sig_herb_control.emit("stop")
            self.reset_herb_runtime_state()
            self.set_chip("label_herbcam_status", "药草 待机", warn=True)

    def on_open_login_page(self):
        self.stop_herb_page_if_needed()
        self.add_log("进入登录认证界面。", "界面")
        self.login_auth.set_running()
        self.stack.setCurrentWidget(self.login_page)
        self.sig_auth_request.emit()

    def on_open_herb_inventory_page(self):
        self.add_log("进入药草出入库界面。", "界面")

        # 进入页面先清空上一次识别缓存，避免模型启动前闪现旧结果。
        self.reset_herb_runtime_state()
        self.herb_inventory_ui.refresh_inventory_table(self.inventory, "--")

        self.stack.setCurrentWidget(self.herb_inventory_page)
        self.herb_page_active = True
        self.sig_herb_control.emit("start")
        self.set_chip("label_herbcam_status", "药草 启动中", warn=True)

    def on_back_to_main(self):
        self.stop_herb_page_if_needed()
        self.add_log("返回主界面。", "界面")
        self.stack.setCurrentWidget(self.main_page)

    def update_time(self):
        self.set_main_label("label_time", f"时间：{now_text()}")

    def add_log(self, text: str, log_type: str = "系统", person: str = "--"):
        line = f"[{now_time()}] {text}"
        plain = self.main_plain("plain_local_log")
        if plain is not None:
            plain.appendPlainText(line)
            cursor = plain.textCursor()
            cursor.movePosition(QTextCursor.End)
            plain.setTextCursor(cursor)
        table = self.main_table("table_system_logs")
        if table is not None:
            table.setUpdatesEnabled(False)
            try:
                row = table.rowCount()
                table.insertRow(row)
                items = [QTableWidgetItem(now_time()), QTableWidgetItem(log_type), QTableWidgetItem(person), QTableWidgetItem(text)]
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row, col, item)
                while table.rowCount() > self.max_log_count:
                    table.removeRow(0)
                table.scrollToBottom()
            finally:
                table.setUpdatesEnabled(True)

    def is_logged_in(self):
        return bool(self.current_user.get("passed", False))

    def is_admin(self):
        return self.is_logged_in() and is_admin_role(self.current_user.get("role", "--"))

    @pyqtSlot(dict)
    def on_login_success(self, data: dict):
        name = str(data.get("name", "--"))
        role = str(data.get("role", "--"))
        score = safe_float(data.get("score", 0.0))
        self.current_user = {"name": name, "role": role, "passed": True, "score": score}
        self.update_auth_display()
        self.update_permission_buttons()
        self.set_realtime_status(f"{name} 登录成功")
        self.set_chip("label_facecam_status", "人脸 已认证")
        self.add_log(f"{name} {role_to_cn(role)} 认证通过，score={score:.3f}", "认证", name)
        QTimer.singleShot(800, self.on_back_to_main)

    @pyqtSlot(dict)
    def on_login_failed(self, data: dict):
        score = safe_float(data.get("score", 0.0))
        name = str(data.get("name", "Unknown"))
        self.current_user = {"name": "--", "role": "--", "passed": False, "score": score}
        self.update_auth_display()
        self.update_permission_buttons()
        self.set_realtime_status("认证失败", alarm=True)
        self.set_chip("label_facecam_status", "人脸 认证失败", warn=True)
        now = time.monotonic()
        if now - self.last_fail_log_time >= 1.0:
            self.last_fail_log_time = now
            self.add_log(f"认证失败：{name}，score={score:.3f}", "认证", "--")

    def on_password_login_demo(self):
        self.login_auth.on_auth_result(json_dumps({"name": "MJN", "role": "admin", "passed": True, "score": 1.0, "status": "passed"}))

    def on_logout_auth(self):
        self.current_user = {"name": "--", "role": "--", "passed": False, "score": 0.0}
        self.update_auth_display()
        self.update_permission_buttons()
        self.set_realtime_status("系统锁定")
        self.set_chip("label_facecam_status", "人脸 待机", warn=True)
        self.login_auth.set_idle()
        self.sig_auth_logout.emit()
        self.add_log("用户注销。", "认证")

    def update_auth_display(self):
        name = self.current_user.get("name", "--")
        role = self.current_user.get("role", "--")
        score = safe_float(self.current_user.get("score", 0.0))
        passed = bool(self.current_user.get("passed", False))
        name_show = name if passed else "未认证"
        role_show = role_to_cn(role) if passed else "--"
        status_show = "已认证" if passed else "锁定"
        score_show = f"{score:.3f}" if passed else "--"
        for label_name, value in [
            ("label_user_name_value", name_show),
            ("label_user_role_value", role_show),
            ("label_auth_status_value", status_show),
            ("label_auth_time_left_value", "--"),
            ("label_auth_result_name_value", name_show),
            ("label_auth_result_role_value", role_show),
            ("label_auth_result_score_value", score_show),
            ("label_auth_node_state_value", status_show),
        ]:
            self.set_main_label(label_name, value)

    @pyqtSlot(QImage)
    def on_herb_frame(self, image: QImage):
        if image.isNull():
            return
        if self.herb_page_active:
            self.herb_inventory_ui.on_frame(image)
        label = self.main_label("label_herb_camera_view")
        if label is not None:
            render_image_to_label(label, image)

    @pyqtSlot(str)
    def on_herb_result(self, text: str):
        data = json_loads_safe(text)
        herb_name = str(data.get("herb_name", data.get("name", "--")))
        confidence = safe_float(data.get("confidence", data.get("score", 0.0)))
        valid = safe_bool(data.get("valid", herb_name in DEFAULT_HERBS))
        if herb_name in ["empty", "none", "空台", "未知", "未放置药材", "低置信度", "--"]:
            valid = False
        old_herb = self.current_herb.get("herb_name", "--")
        old_conf = safe_float(self.current_herb.get("confidence", 0.0))
        old_valid = bool(self.current_herb.get("valid", False))
        if old_herb == herb_name and abs(old_conf - confidence) < 0.001 and old_valid == valid:
            return
        self.current_herb = {"herb_name": herb_name, "confidence": confidence, "valid": valid}
        self.update_herb_display()
        self.update_permission_buttons()
        self.herb_inventory_ui.update_recognition(herb_name, confidence, valid)
        self.herb_inventory_ui.refresh_inventory_table(self.inventory, herb_name if valid else "--")
        if valid:
            self.set_chip("label_herbcam_status", "药草 在线")
            self.set_realtime_status(f"识别：{herb_name}")
        else:
            self.set_chip("label_herbcam_status", "药草 无效", warn=True)

    def update_herb_display(self):
        herb = self.current_herb.get("herb_name", "--")
        confidence = safe_float(self.current_herb.get("confidence", 0.0))
        valid = bool(self.current_herb.get("valid", False))
        herb_show = herb if valid else "--"
        conf_show = f"{confidence:.3f}" if valid else "--"
        for label_name, value in [
            ("label_current_herb_value", herb_show),
            ("label_herb_confidence_value", conf_show),
            ("label_page_herb_name_value", herb_show),
            ("label_page_conf_value", conf_show),
        ]:
            self.set_main_label(label_name, value)

    @pyqtSlot(str)
    def on_weight(self, text: str):
        data = json_loads_safe(text)
        weight = safe_float(data.get("weight", 0.0))
        stable = safe_bool(data.get("stable", False))
        unit = str(data.get("unit", "g"))
        old_weight = safe_float(self.current_weight.get("weight", 0.0))
        old_stable = bool(self.current_weight.get("stable", False))
        old_unit = self.current_weight.get("unit", "g")
        if abs(old_weight - weight) < 0.05 and old_stable == stable and old_unit == unit:
            return
        self.current_weight = {"weight": weight, "stable": stable, "unit": unit}
        self.update_weight_display()
        self.update_permission_buttons()
        self.herb_inventory_ui.update_weight(weight, stable, unit)
        self.set_chip("label_hx711_status", "HX711 稳定" if stable else "HX711 波动", warn=not stable)

    def update_weight_display(self):
        weight = safe_float(self.current_weight.get("weight", 0.0))
        stable = bool(self.current_weight.get("stable", False))
        unit = self.current_weight.get("unit", "g")
        weight_text = f"{weight:.1f} {unit}"
        stable_text = "稳定" if stable else "未稳定"
        for label_name, value in [
            ("label_current_weight_value", weight_text),
            ("label_weight_stable_value", stable_text),
            ("label_page_weight_value", weight_text),
            ("label_page_stable_value", "是" if stable else "否"),
        ]:
            self.set_main_label(label_name, value)

    @pyqtSlot(str)
    def on_env_status(self, text: str):
        data = json_loads_safe(text)
        temperature = safe_float(data.get("temperature", data.get("temp", 0.0)))
        humidity = safe_float(data.get("humidity", data.get("hum", 0.0)))
        env_status = str(data.get("env_status", data.get("status", "normal")))
        advice = str(data.get("action_suggestion", data.get("advice", "保持")))
        sensor = str(data.get("sensor", "AHT30"))
        temp_text = f"{temperature:.1f} ℃"
        hum_text = f"{humidity:.1f} %"
        for label_name, value in [
            ("label_temperature_value", temp_text),
            ("label_humidity_value", hum_text),
            ("label_control_advice_value", advice),
            ("label_env_page_temp_value", temp_text),
            ("label_env_page_hum_value", hum_text),
            ("label_env_page_advice_value", advice),
        ]:
            self.set_main_label(label_name, value)
        if env_status in ["error", "offline", "读取失败"]:
            self.set_chip("label_aht20_status", f"{sensor} 异常", alarm=True)
            self.set_realtime_status("环境传感器异常", alarm=True)
        elif env_status in ["danger", "alarm", "危险"]:
            self.set_chip("label_aht20_status", f"{sensor} 危险", alarm=True)
            self.set_realtime_status("环境危险", alarm=True)
        elif env_status in ["hot", "humid", "warn", "cold", "dry", "高温", "湿度偏高"]:
            self.set_chip("label_aht20_status", f"{sensor} 警告", warn=True)
            self.set_realtime_status(advice)
        else:
            self.set_chip("label_aht20_status", f"{sensor} 正常")
            self.set_realtime_status("环境正常")

    def on_inventory_action(self, action: str):
        if not self.is_logged_in():
            self.set_realtime_status("请先登录认证", alarm=True)
            self.add_log("出入库被拒绝：未认证。", "库存")
            return
        herb = self.current_herb.get("herb_name", "--")
        valid = bool(self.current_herb.get("valid", False))
        if not valid or herb not in DEFAULT_HERBS:
            self.set_realtime_status("药草识别无效", alarm=True)
            self.add_log("出入库被拒绝：药草识别无效。", "库存")
            return
        weight = safe_float(self.current_weight.get("weight", 0.0))
        stable = bool(self.current_weight.get("stable", False))
        if not stable or weight <= 0:
            self.set_realtime_status("称重未稳定", alarm=True)
            self.add_log("出入库被拒绝：称重未稳定。", "库存")
            return
        stock = safe_float(self.inventory[herb]["weight"])
        if action == "stock_out" and weight > stock:
            self.set_realtime_status("库存不足", alarm=True)
            self.add_log(f"出库失败：{herb} 库存不足。", "库存")
            return
        if action == "stock_in":
            self.inventory[herb]["weight"] = stock + weight
            action_text = "入库"
        else:
            self.inventory[herb]["weight"] = stock - weight
            action_text = "出库"
        self.refresh_inventory_table()
        payload = {
            "herb_name": herb,
            "weight": weight,
            "operator": self.current_user.get("name", "--"),
            "role": self.current_user.get("role", "--"),
            "confidence": self.current_herb.get("confidence", 0.0),
        }
        self.sig_inventory_action.emit(action, json_dumps(payload))
        self.set_realtime_status(f"{action_text}完成：{herb}")
        self.set_main_label("label_last_stock_action_value", f"{action_text} {herb}")
        self.add_log(f"{action_text}确认：{herb} {weight:.1f}g", "库存", payload["operator"])

    @pyqtSlot(str)
    def on_inventory_status(self, text: str):
        data = json_loads_safe(text)
        items = data.get("items", [])
        if isinstance(items, list):
            for item in items:
                herb = str(item.get("herb_name", item.get("name", "")))
                if herb not in self.inventory:
                    continue
                self.inventory[herb]["weight"] = safe_float(item.get("weight", 0.0))
                self.inventory[herb]["threshold"] = safe_float(item.get("threshold", 50.0))
                self.inventory[herb]["status"] = str(item.get("status", "正常"))
        last_action = str(data.get("last_action", ""))
        if last_action:
            self.set_main_label("label_last_stock_action_value", last_action)
        self.refresh_inventory_table()

    def refresh_inventory_table(self):
        table = self.main_table("table_inventory")
        low_items = []
        for herb in DEFAULT_HERBS:
            info = self.inventory[herb]
            weight = safe_float(info.get("weight", 0.0))
            threshold = safe_float(info.get("threshold", 50.0))
            if weight < threshold:
                info["status"] = "低库存"
                low_items.append(herb)
            else:
                info["status"] = "正常"
        if table is not None:
            current_herb = self.current_herb.get("herb_name", "--")
            highlight_brush = QBrush(QColor(29, 78, 58))
            normal_brush = QBrush()

            table.setUpdatesEnabled(False)
            try:
                table.setRowCount(len(DEFAULT_HERBS))
                for row, herb in enumerate(DEFAULT_HERBS):
                    info = self.inventory[herb]
                    values = [herb, f"{safe_float(info['weight']):.1f}", f"{safe_float(info['threshold']):.1f}", str(info["status"])]
                    is_current = current_herb == herb

                    for col, value in enumerate(values):
                        item = table.item(row, col)
                        if item is None:
                            item = QTableWidgetItem(value)
                            item.setTextAlignment(Qt.AlignCenter)
                            table.setItem(row, col, item)
                        elif item.text() != value:
                            item.setText(value)

                        item.setBackground(highlight_brush if is_current else normal_brush)
            finally:
                table.setUpdatesEnabled(True)
        self.set_main_label("label_herb_kind_count_value", str(len(DEFAULT_HERBS)))
        self.set_main_label("label_low_stock_count_value", str(len(low_items)))
        self.set_main_label("label_low_stock_items_value", "、".join(low_items) if low_items else "无")
        self.herb_inventory_ui.refresh_inventory_table(self.inventory, self.current_herb.get("herb_name", "--"))

    def update_permission_buttons(self):
        logged = self.is_logged_in()
        admin = self.is_admin()
        herb_valid = bool(self.current_herb.get("valid", False))
        weight_stable = bool(self.current_weight.get("stable", False))
        weight_value = safe_float(self.current_weight.get("weight", 0.0))
        stock_allowed = logged and herb_valid and weight_stable and weight_value > 0
        self.set_button_enabled("btn_stock_in", stock_allowed)
        self.set_button_enabled("btn_stock_out", stock_allowed)
        self.herb_inventory_ui.set_buttons_enabled(stock_allowed)
        check = self.main_check("check_gate_override")
        override = check.isChecked() if check is not None else False
        gate_allowed = admin and override
        self.set_button_enabled("btn_gate_open", gate_allowed)
        self.set_button_enabled("btn_gate_close", gate_allowed)
        for name in ["btn_admin_add_user", "btn_admin_role", "btn_admin_inventory", "btn_admin_threshold"]:
            self.set_button_enabled(name, admin)

    def on_gate_cmd(self, cmd: str):
        if not self.is_admin():
            self.set_realtime_status("权限不足", alarm=True)
            self.add_log("大门控制被拒绝：需要管理员权限。", "控制")
            return
        operator = self.current_user.get("name", "--")
        self.sig_door_cmd.emit(cmd, "admin_manual", operator)
        self.set_realtime_status(f"大门命令：{cmd}")
        self.add_log(f"大门命令：{cmd}", "控制", operator)

    def on_device_cmd(self, device: str, cmd: str, reason: str):
        operator = self.current_user.get("name", "--")
        self.sig_device_cmd.emit(device, cmd, reason, operator)
        self.set_realtime_status(f"{device} {cmd}")
        self.add_log(f"设备命令：{device} -> {cmd}", "控制", operator)

    @pyqtSlot(str)
    def on_door_state(self, text: str):
        data = json_loads_safe(text)
        state = str(data.get("state", data.get("door_state", "closed")))
        if state in ["open", "opened", "打开"]:
            self.set_main_label("label_gate_state_value", "打开")
            self.set_main_label("label_gate_output_value", "打开")
            self.set_chip("label_gate_servo_status", "舵机 打开", warn=True)
        else:
            self.set_main_label("label_gate_state_value", "关闭")
            self.set_main_label("label_gate_output_value", "关闭")
            self.set_chip("label_gate_servo_status", "舵机 关闭", warn=True)

    @pyqtSlot(str)
    def on_device_status(self, text: str):
        data = json_loads_safe(text)
        device = str(data.get("device", ""))
        state = str(data.get("state", data.get("cmd", "")))
        if device in ["face_camera", "facecam"]:
            show = "在线" if state in ["online", "running", "on"] else "离线"
            self.set_chip("label_facecam_status", f"人脸 {show}", warn=(show != "在线"))
        elif device in ["herb_camera", "herbcam"]:
            if state == "standby":
                self.set_chip("label_herbcam_status", "药草 待机", warn=True)
            else:
                show = "在线" if state in ["online", "running", "on"] else "离线"
                self.set_chip("label_herbcam_status", f"药草 {show}", warn=(show != "在线"))
        elif device in ["aht20", "aht30", "aht30_sensor", "aht30_node"]:
            show = "在线" if state in ["online", "running", "on", "正常"] else "离线"
            sensor = str(data.get("sensor", "AHT30"))
            self.set_chip("label_aht20_status", f"{sensor} {show}", warn=(show != "在线"))
        elif device == "fan":
            show = "开启" if state in ["on", "start", "开启"] else "待命"
            self.set_main_label("label_fan_output_value", show)
            self.set_chip("label_fan_status", f"风扇 {show}", warn=(show == "待命"))
        elif device == "alarm":
            show = "开启" if state in ["on", "start", "开启"] else "待命"
            self.set_main_label("label_alarm_output_value", show)
            self.set_chip("label_alarm_status", f"声光 {show}", alarm=(show == "开启"), warn=(show == "待命"))

    def on_admin_action(self, button_name: str):
        if not self.is_admin():
            self.set_realtime_status("权限不足", alarm=True)
            self.add_log("管理操作被拒绝：需要管理员权限。", "管理")
            return
        mapping = {"btn_admin_add_user": "新增人员", "btn_admin_role": "权限管理", "btn_admin_inventory": "库存修正", "btn_admin_threshold": "阈值设置"}
        detail = mapping.get(button_name, button_name)
        self.set_realtime_status(detail)
        self.add_log(f"管理员操作：{detail}", "管理", self.current_user.get("name", "--"))
        self.sig_ui_event.emit("admin_action", detail)

    @pyqtSlot(str)
    def on_system_event(self, text: str):
        data = json_loads_safe(text)
        detail = str(data.get("detail", data.get("raw", text)))
        level = str(data.get("level", "info"))
        self.add_log(detail, "事件")
        if level in ["error", "alarm", "danger"]:
            self.set_realtime_status(detail, alarm=True)
        else:
            self.set_realtime_status(detail)

    def on_exit_system(self):
        self.request_shutdown("退出系统按钮")

    def request_shutdown(self, reason: str):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.stop_herb_page_if_needed()
        self.add_log(f"收到关闭请求：{reason}", "系统")
        self.add_log("正在安全停止 ROS2 节点。", "系统")
        self.set_realtime_status("系统退出中", alarm=True)
        self.disable_buttons_for_shutdown()
        self.sig_ui_event.emit("shutdown", reason)
        self.sig_request_ros_stop.emit()
        QTimer.singleShot(3000, self.force_close_after_timeout)

    def disable_buttons_for_shutdown(self):
        names = [
            "btn_home_overview", "btn_lock_toggle", "btn_herb_inventory", "btn_env_control", "btn_log_trace",
            "btn_admin_settings", "btn_exit_system", "btn_start_auth", "btn_logout_auth", "btn_stock_in",
            "btn_stock_out", "btn_gate_open", "btn_gate_close", "btn_fan_start", "btn_fan_stop",
            "btn_alarm_start", "btn_alarm_stop",
        ]
        for name in names:
            self.set_button_enabled(name, False)
        self.herb_inventory_ui.set_buttons_enabled(False)

    @pyqtSlot()
    def on_ros_worker_finished(self):
        self.add_log("ROS2 后台线程已停止。", "系统")
        self.allow_close = True
        QTimer.singleShot(0, self.close)

    def force_close_after_timeout(self):
        if self.allow_close:
            return
        self.add_log("ROS2 后台线程停止超时，执行兜底关闭。", "系统")
        self.allow_close = True
        self.close()

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
            return
        event.ignore()
        self.request_shutdown("窗口关闭事件")


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    window = MainWindow()
    ros_thread = QThread()
    ros_worker = RosWorker()
    ros_worker.moveToThread(ros_thread)
    ros_thread.started.connect(ros_worker.start)
    window.connect_ros_worker(ros_worker)
    ros_worker.finished.connect(window.on_ros_worker_finished)
    ros_worker.finished.connect(ros_thread.quit)
    ros_thread.start()

    def handle_os_signal(signum, frame):
        reason = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else f"signal={signum}"
        QTimer.singleShot(0, lambda: window.request_shutdown(reason))

    signal.signal(signal.SIGTERM, handle_os_signal)
    signal.signal(signal.SIGINT, handle_os_signal)

    window.showFullScreen()
    exit_code = app.exec_()

    try:
        if ros_thread is not None and ros_thread.isRunning():
            try:
                QMetaObject.invokeMethod(ros_worker, "stop", Qt.BlockingQueuedConnection)
            except RuntimeError:
                pass
            ros_thread.quit()
            ros_thread.wait(2000)
    except RuntimeError:
        pass

    try:
        ros_worker.deleteLater()
    except RuntimeError:
        pass

    try:
        ros_thread.deleteLater()
    except RuntimeError:
        pass

    if rclpy.ok():
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
