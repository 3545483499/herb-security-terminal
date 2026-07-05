#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aws_iot_bridge.py

AWS IoT Core ↔ ROS2 桥接节点

职责：
1. 通过 MQTT（paho-mqtt）连接到 AWS IoT Core
2. 订阅云端命令 → 发布到 ROS2 /cloud/cmd
3. 订阅 ROS2 /cloud/telemetry → 上报到 AWS IoT

这样其他 ROS 节点不需要知道 AWS，只需用标准 ROS topic 通信。
"""

import sys
import site
import json
import time
import ssl
import threading
from pathlib import Path
from typing import Optional

# ============================================================
# 虚拟环境 site-packages（同 login_auth_node）
# ============================================================

VENV_SITE = Path.home() / "venvs" / "ROS_herb_robot" / "lib" / "python3.12" / "site-packages"

if VENV_SITE.exists():
    site.addsitedir(str(VENV_SITE))
    sys.path.insert(0, str(VENV_SITE))

import paho.mqtt.client as mqtt

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ============================================================
# AWS IoT Bridge Node
# ============================================================

class AwsIotBridge(Node):
    """
    AWS IoT MQTT ↔ ROS2 桥接节点

    参数（来自 aws_iot_bridge.yaml）：
      - endpoint       : AWS IoT endpoint（如 xxx-ats.iot.ap-southeast-2.amazonaws.com）
      - client_id      : MQTT client_id（如 K1_HerbRobot）
      - cert_dir       : 证书目录路径
      - ca_file        : 根证书文件名（如 AmazonRootCA1.pem）
      - cert_file      : 设备证书文件名（如 device.pem.crt）
      - key_file       : 私钥文件名（如 private.pem.key）
      - port           : MQTT 端口，默认 8883
      - cloud_cmd_topic    : 云端下发的命令 topic
      - telemetry_topic    : 本地上报数据的 topic
      - reconnect_delay    : 断线重连间隔（秒）
      - keepalive          : MQTT keepalive（秒）
    """

    def __init__(self):
        super().__init__("aws_iot_bridge")

        # ---- 声明参数 ----
        self.declare_parameter("endpoint", "")
        self.declare_parameter("client_id", "K1_HerbRobot")
        self.declare_parameter("cert_dir", str(Path.home() / "herbsecurity_ws" / "aws_iot" / "certs"))
        self.declare_parameter("ca_file", "AmazonRootCA1.pem")
        self.declare_parameter("cert_file", "device.pem.crt")
        self.declare_parameter("key_file", "private.pem.key")
        self.declare_parameter("port", 8883)
        self.declare_parameter("cloud_cmd_topic", "herb_robot/cloud/cmd")
        self.declare_parameter("telemetry_topic", "herb_robot/telemetry")
        self.declare_parameter("reconnect_delay", 5.0)
        self.declare_parameter("keepalive", 30)

        # ---- 读取参数 ----
        self.endpoint = str(self.get_parameter("endpoint").value)
        self.client_id = str(self.get_parameter("client_id").value)
        self.cert_dir = Path(str(self.get_parameter("cert_dir").value)).expanduser()
        self.ca_file = str(self.get_parameter("ca_file").value)
        self.cert_file = str(self.get_parameter("cert_file").value)
        self.key_file = str(self.get_parameter("key_file").value)
        self.port = int(self.get_parameter("port").value)
        self.cloud_cmd_topic = str(self.get_parameter("cloud_cmd_topic").value)
        self.telemetry_topic = str(self.get_parameter("telemetry_topic").value)
        self.reconnect_delay = float(self.get_parameter("reconnect_delay").value)
        self.keepalive = int(self.get_parameter("keepalive").value)

        # ---- 状态 ----
        self.connected = False
        self.mqtt_client: Optional[mqtt.Client] = None
        self.stop_flag = threading.Event()

        # ---- ROS2 发布/订阅 ----
        # 云端命令 → ROS topic
        self.cloud_cmd_pub = self.create_publisher(String, "/cloud/cmd", 10)

        # 设备状态 → 云端
        self.telemetry_sub = self.create_subscription(
            String,
            "/cloud/telemetry",
            self.on_telemetry,
            10,
        )

        # 也监听系统事件
        self.system_event_sub = self.create_subscription(
            String,
            "/system/event",
            self.on_system_event,
            10,
        )

        # ---- 启动 MQTT ----
        self.init_mqtt()

    # =====================================================
    # MQTT 初始化
    # =====================================================

    def init_mqtt(self):
        """创建 MQTT 客户端，配置 TLS 证书，连接 AWS IoT"""
        if not self.endpoint:
            self.get_logger().error("未配置 AWS IoT endpoint，桥接节点不启动 MQTT")
            return

        # 证书路径
        ca_path = str(self.cert_dir / self.ca_file)
        cert_path = str(self.cert_dir / self.cert_file)
        key_path = str(self.cert_dir / self.key_file)

        # 检查证书
        for p, name in [(ca_path, "CA"), (cert_path, "设备证书"), (key_path, "私钥")]:
            if not Path(p).exists():
                self.get_logger().error(f"缺少 {name} 文件：{p}")
                return

        # 创建客户端（MQTT v3.1.1，AWS IoT 标准）
        self.mqtt_client = mqtt.Client(
            client_id=self.client_id,
            protocol=mqtt.MQTTv311,
        )

        # TLS 配置
        self.mqtt_client.tls_set(
            ca_certs=ca_path,
            certfile=cert_path,
            keyfile=key_path,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

        # 回调
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message

        # 连接（在后台线程中）
        try:
            self.get_logger().info(f"正在连接 AWS IoT: {self.endpoint}:{self.port}")
            self.mqtt_client.connect(self.endpoint, port=self.port, keepalive=self.keepalive)

            # 启动网络循环线程
            self.mqtt_client.loop_start()

        except Exception as e:
            self.get_logger().error(f"AWS IoT 连接失败: {e}")

    # =====================================================
    # MQTT 回调
    # =====================================================

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT 连接成功回调"""
        if rc == 0:
            self.connected = True
            self.get_logger().info("✓ 已连接到 AWS IoT Core")

            # 订阅云端命令 topic
            client.subscribe(self.cloud_cmd_topic, qos=1)
            self.get_logger().info(f"已订阅云端命令: {self.cloud_cmd_topic}")

            # 发布上线消息
            self._publish_mqtt(
                f"{self.telemetry_topic}/status",
                json.dumps({
                    "type": "online",
                    "client_id": self.client_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }),
            )

        else:
            self.get_logger().error(f"AWS IoT 连接失败，rc={rc}")
            self.connected = False

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT 断线回调，自动重连由 paho loop 处理"""
        self.connected = False
        self.get_logger().warn(f"AWS IoT 连接断开，rc={rc}，将自动重连")

    def _on_mqtt_message(self, client, userdata, msg):
        """收到云端消息 → 转发到 ROS2 topic"""
        try:
            payload = msg.payload.decode("utf-8")
            self.get_logger().info(f"云端消息 [{msg.topic}]: {payload[:200]}")

            ros_msg = String()
            ros_msg.data = payload
            self.cloud_cmd_pub.publish(ros_msg)

        except Exception as e:
            self.get_logger().error(f"处理云端消息异常: {e}")

    # =====================================================
    # ROS2 订阅回调 → MQTT 上报
    # =====================================================

    def on_telemetry(self, msg: String):
        """本地遥测数据 → 上报到 AWS IoT"""
        self._publish_mqtt(self.telemetry_topic, msg.data)

    def on_system_event(self, msg: String):
        """系统事件 → 上报到 AWS IoT"""
        self._publish_mqtt(f"{self.telemetry_topic}/event", msg.data)

    # =====================================================
    # 内部方法
    # =====================================================

    def _publish_mqtt(self, topic: str, payload: str):
        """安全发布 MQTT 消息"""
        if self.mqtt_client is None or not self.connected:
            return

        try:
            self.mqtt_client.publish(topic, payload, qos=1)
        except Exception as e:
            self.get_logger().error(f"MQTT 发布失败 [{topic}]: {e}")

    # =====================================================
    # 销毁
    # =====================================================

    def destroy_node(self):
        self.get_logger().info("正在关闭 AWS IoT 桥接节点...")

        # 发送离线消息
        if self.connected and self.mqtt_client:
            try:
                self._publish_mqtt(
                    f"{self.telemetry_topic}/status",
                    json.dumps({
                        "type": "offline",
                        "client_id": self.client_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }),
                )
            except Exception:
                pass

        # 停止 MQTT
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.mqtt_client = None

        self.stop_flag.set()
        super().destroy_node()


# ============================================================
# main
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        print("========== AWS IoT Bridge ROS2 Node ==========")

        node = AwsIotBridge()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出 AWS IoT Bridge。")

    except Exception as e:
        print("AWS IoT Bridge 异常:", repr(e))

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("AWS IoT Bridge 已退出。")


if __name__ == "__main__":
    main()
