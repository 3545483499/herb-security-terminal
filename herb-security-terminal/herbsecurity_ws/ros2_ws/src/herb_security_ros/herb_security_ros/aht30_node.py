#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
env_aht30_node.py

AHT30 温湿度 ROS2 独立节点

功能：
1. 通过 I2C 读取 AHT30 温湿度
2. 发布 /env/status 给 MainWindow.py 实时显示
3. 发布 /device/status 给主界面顶部设备状态
4. 发布 /system/event 写入系统日志

主界面接收字段：
- temperature
- humidity
- env_status
- action_suggestion
"""

import sys
import site
import json
import time
from pathlib import Path

# ============================================================
# 让 /usr/bin/python3 启动 ROS2 console_scripts 时也能找到 venv 库
# ============================================================

VENV_SITE = Path.home() / "venvs" / "ROS_herb_robot" / "lib" / "python3.12" / "site-packages"

if VENV_SITE.exists():
    site.addsitedir(str(VENV_SITE))
    sys.path.insert(0, str(VENV_SITE))

from smbus2 import SMBus, i2c_msg

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ============================================================
# 工具函数
# ============================================================

def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def crc8_aht30(data):
    """
    AHT30 CRC8 校验
    多项式: 0x31
    初始值: 0xFF
    """
    crc = 0xFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


# ============================================================
# AHT30 读取类
# ============================================================

class AHT30Reader:
    def __init__(self, bus_num: int, address: int):
        self.bus_num = bus_num
        self.address = address
        self.bus = None

    def open(self):
        if self.bus is not None:
            return

        self.bus = SMBus(self.bus_num)

    def close(self):
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass

            self.bus = None

    def read_once(self):
        """
        读取 AHT30 温湿度
        返回：
            temperature: ℃
            humidity: %RH
        """
        if self.bus is None:
            self.open()

        # 触发一次测量：0xAC 0x33 0x00
        write_msg = i2c_msg.write(self.address, [0xAC, 0x33, 0x00])
        self.bus.i2c_rdwr(write_msg)

        # 等待测量完成
        time.sleep(0.08)

        # 读取 7 字节
        read_msg = i2c_msg.read(self.address, 7)
        self.bus.i2c_rdwr(read_msg)

        data = list(read_msg)

        if len(data) != 7:
            raise RuntimeError(f"AHT30 数据长度异常: len={len(data)}, raw={data}")

        # data[0] bit7 = 1 表示忙
        if data[0] & 0x80:
            raise RuntimeError(f"AHT30 正忙，raw={data}")

        crc_calc = crc8_aht30(data[0:6])
        crc_recv = data[6]

        if crc_calc != crc_recv:
            raise RuntimeError(
                f"CRC 校验失败: calc=0x{crc_calc:02X}, "
                f"recv=0x{crc_recv:02X}, raw={data}"
            )

        # 湿度原始值：20 bit
        hum_raw = (
            (data[1] << 12)
            | (data[2] << 4)
            | (data[3] >> 4)
        ) & 0xFFFFF

        # 温度原始值：20 bit
        temp_raw = (
            ((data[3] & 0x0F) << 16)
            | (data[4] << 8)
            | data[5]
        ) & 0xFFFFF

        humidity = hum_raw * 100.0 / 1048576.0
        temperature = temp_raw * 200.0 / 1048576.0 - 50.0

        return temperature, humidity


# ============================================================
# ROS2 节点
# ============================================================

class EnvAHT30Node(Node):
    def __init__(self):
        super().__init__("env_aht30_node")

        # I2C 参数
        self.declare_parameter("bus_num", 3)
        self.declare_parameter("address", 0x38)
        self.declare_parameter("period", 1.0)

        # 环境阈值
        self.declare_parameter("temp_high", 30.0)
        self.declare_parameter("temp_low", 10.0)
        self.declare_parameter("humidity_high", 70.0)
        self.declare_parameter("humidity_low", 35.0)

        self.bus_num = int(self.get_parameter("bus_num").value)
        self.address = int(self.get_parameter("address").value)
        self.period = float(self.get_parameter("period").value)

        self.temp_high = float(self.get_parameter("temp_high").value)
        self.temp_low = float(self.get_parameter("temp_low").value)
        self.humidity_high = float(self.get_parameter("humidity_high").value)
        self.humidity_low = float(self.get_parameter("humidity_low").value)

        self.reader = AHT30Reader(
            bus_num=self.bus_num,
            address=self.address,
        )

        # 发布给主界面
        self.env_pub = self.create_publisher(String, "/env/status", 10)

        # 发布设备状态
        self.device_status_pub = self.create_publisher(String, "/device/status", 10)

        # 发布系统事件
        self.system_event_pub = self.create_publisher(String, "/system/event", 10)

        self.fail_count = 0
        self.last_env_status = ""

        self.timer = self.create_timer(self.period, self.on_timer)

        self.publish_system_event(
            f"AHT30 环境节点启动: i2c-{self.bus_num}, addr=0x{self.address:02X}",
            "info",
        )

        self.publish_device_status("online")

        self.get_logger().info(
            f"AHT30 环境节点启动: i2c-{self.bus_num}, "
            f"addr=0x{self.address:02X}, period={self.period}s"
        )

    # =====================================================
    # 环境判断
    # =====================================================

    def judge_env(self, temperature: float, humidity: float):
        """
        返回：
            env_status:
                normal / hot / cold / humid / dry / danger
            advice:
                主界面显示建议
        """

        if temperature >= self.temp_high and humidity >= self.humidity_high:
            return "danger", "高温高湿，建议开启风扇并报警"

        if temperature >= self.temp_high:
            return "hot", "温度偏高，建议开启风扇"

        if temperature <= self.temp_low:
            return "cold", "温度偏低，建议关闭风扇并保温"

        if humidity >= self.humidity_high:
            return "humid", "湿度偏高，建议通风除湿"

        if humidity <= self.humidity_low:
            return "dry", "湿度偏低，建议适当加湿"

        return "normal", "环境正常，保持当前状态"

    # =====================================================
    # 发布函数
    # =====================================================

    def publish_env_status(self, temperature: float, humidity: float):
        env_status, advice = self.judge_env(temperature, humidity)

        msg = String()
        msg.data = json_dumps(
            {
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 2),

                # 兼容 MainWindow.py 里的备用字段读取
                "temp": round(temperature, 2),
                "hum": round(humidity, 2),

                "env_status": env_status,
                "status": env_status,

                "action_suggestion": advice,
                "advice": advice,

                "sensor": "AHT30",
                "bus_num": self.bus_num,
                "address": f"0x{self.address:02X}",
                "timestamp": now_text(),
            }
        )

        self.env_pub.publish(msg)

        self.publish_device_status("online")

        if env_status != self.last_env_status:
            self.last_env_status = env_status

            level = "info"
            if env_status in ["danger"]:
                level = "error"
            elif env_status in ["hot", "cold", "humid", "dry"]:
                level = "warn"

            self.publish_system_event(
                f"AHT30 环境状态: {env_status}, "
                f"温度={temperature:.2f}℃, 湿度={humidity:.2f}%RH, 建议={advice}",
                level,
            )

    def publish_device_status(self, state: str, error: str = ""):
        """
        这里 device 同时发 aht20 和 aht30。
        原因：你主界面标签目前叫 label_aht20_status，
        但传感器实际是 AHT30。
        """

        for device_name in ["aht20", "aht30"]:
            msg = String()
            msg.data = json_dumps(
                {
                    "device": device_name,
                    "state": state,
                    "sensor": "AHT30",
                    "error": error,
                    "timestamp": now_text(),
                }
            )
            self.device_status_pub.publish(msg)

    def publish_system_event(self, detail: str, level: str = "info"):
        msg = String()
        msg.data = json_dumps(
            {
                "type": "env",
                "detail": detail,
                "level": level,
                "timestamp": now_text(),
            }
        )
        self.system_event_pub.publish(msg)

    def publish_error_status(self, error: str):
        self.publish_device_status("offline", error)

        msg = String()
        msg.data = json_dumps(
            {
                "temperature": 0.0,
                "humidity": 0.0,
                "temp": 0.0,
                "hum": 0.0,

                "env_status": "error",
                "status": "error",

                "action_suggestion": "AHT30 读取失败，请检查 I2C 接线",
                "advice": "AHT30 读取失败，请检查 I2C 接线",

                "sensor": "AHT30",
                "bus_num": self.bus_num,
                "address": f"0x{self.address:02X}",
                "error": error,
                "timestamp": now_text(),
            }
        )

        self.env_pub.publish(msg)

    # =====================================================
    # 定时读取
    # =====================================================

    def on_timer(self):
        try:
            temperature, humidity = self.reader.read_once()

            self.fail_count = 0

            self.publish_env_status(temperature, humidity)

            self.get_logger().info(
                f"AHT30: 温度={temperature:.2f}℃ 湿度={humidity:.2f}%RH"
            )

        except Exception as e:
            self.fail_count += 1
            error = str(e)

            self.get_logger().error(
                f"AHT30 读取失败({self.fail_count}): {error}"
            )

            # 第一次失败立即上报，之后每 5 次上报一次，避免刷屏
            if self.fail_count == 1 or self.fail_count % 5 == 0:
                self.publish_error_status(error)
                self.publish_system_event(
                    f"AHT30 读取失败: {error}",
                    "error",
                )

            # I2C 异常后重开，避免总线卡死
            try:
                self.reader.close()
            except Exception:
                pass

    # =====================================================
    # 销毁
    # =====================================================

    def destroy_node(self):
        self.get_logger().info("正在关闭 AHT30 环境节点...")

        try:
            self.reader.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        print("========== Env AHT30 ROS2 Node ==========")

        node = EnvAHT30Node()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出 env_aht30_node。")

    except Exception as e:
        print("env_aht30_node 异常:", repr(e))

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("env_aht30_node 已退出。")


if __name__ == "__main__":
    main()