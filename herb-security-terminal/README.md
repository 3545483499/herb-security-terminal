# Herb Security Terminal / 中药材识别与仓储终端

本仓库用于保存 **MUSE Pi Pro / K1** 端中药材识别、登录认证、温湿度采集、云端通信和主界面程序。代码主体是 ROS 2 Python 包 `herb_security_ros`。

## 项目功能

- PyQt5 主界面：登录、人脸认证、库存/出入库、设备状态显示。
- ROS 2 节点：
  - `MainWindow`：主界面入口。
  - `login_auth_node`：摄像头人脸登录认证。
  - `herb_recognition_node`：药材图像识别。
  - `aht30_node`：AHT30 温湿度采集。
  - `aws_iot_bridge`：AWS IoT Core 与 ROS 2 topic 桥接。
- 独立脚本：
  - `read_HX711.py`：HX711 称重读取测试。
  - `run_gouqi_yolo_fp32_k1.py`：K1 端 ONNXRuntime / SpaceMIT EP 推理测试。

## 仓库目录

```text
herbsecurity_ws/
├── read_HX711.py
├── run_gouqi_yolo_fp32_k1.py
├── aws_iot/
│   └── certs/              # 证书目录，本仓库不提交真实证书
├── face_data/              # 人脸库目录，本仓库不提交 face_database.npz
├── models/                 # 模型目录，本仓库不提交 .onnx 大模型
│   ├── buffalo_s/
│   ├── gouqi_yolo/
│   └── herb_type/
└── ros2_ws/
    └── src/
        └── herb_security_ros/
            ├── config/
            ├── herb_security_ros/
            ├── launch/
            ├── ui/
            ├── package.xml
            └── setup.py
```

## 重要说明：模型和证书没有放进仓库

原始 zip 中含有 AWS IoT 设备证书、私钥和多个 ONNX 模型。为了避免泄露密钥以及超过 GitHub 单文件限制，本整理版已经移除：

- `aws_iot/certs/*.pem`
- `aws_iot/certs/*.crt`
- `aws_iot/certs/*.key`
- `face_data/face_database.npz`
- `models/**/*.onnx`
- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`
- `__pycache__/`

部署到 K1 时，需要手动把模型和证书放回对应目录。

## 运行环境

建议环境：

- Ubuntu / Debian 系 Linux
- ROS 2
- Python 3.12
- PyQt5
- OpenCV
- NumPy
- ONNXRuntime / SpaceMIT 版 ONNXRuntime
- paho-mqtt
- smbus2
- python3-libgpiod / gpiod

可先安装 Python 侧依赖：

```bash
pip install -r requirements.txt
```

注意：`rclpy`、`std_msgs`、`sensor_msgs`、`launch_ros` 通常来自 ROS 2，不建议用普通 pip 安装。

## 模型文件放置位置

整理版不含 ONNX 模型。部署前按下面路径放回：

```text
/home/mjn/herbsecurity_ws/models/buffalo_s/det_500m.onnx
/home/mjn/herbsecurity_ws/models/buffalo_s/w600k_mbf.onnx
/home/mjn/herbsecurity_ws/models/herb_type/herb_type_mbv3_160_fp32.onnx
/home/mjn/herbsecurity_ws/models/gouqi_yolo/gouqi_yolo_int8.onnx
/home/mjn/herbsecurity_ws/face_data/face_database.npz
```

如果实际路径不是 `/home/mjn/herbsecurity_ws`，需要同步修改：

```text
herbsecurity_ws/ros2_ws/src/herb_security_ros/config/login_auth.yaml
herbsecurity_ws/ros2_ws/src/herb_security_ros/config/herb_recognition.yaml
herbsecurity_ws/ros2_ws/src/herb_security_ros/config/aws_iot_bridge.yaml
herbsecurity_ws/run_gouqi_yolo_fp32_k1.py
```

## AWS IoT 证书放置位置

本仓库不提交真实证书。部署时手动放入：

```text
/home/mjn/herbsecurity_ws/aws_iot/certs/AmazonRootCA1.pem
/home/mjn/herbsecurity_ws/aws_iot/certs/device.pem.crt
/home/mjn/herbsecurity_ws/aws_iot/certs/private.pem.key
```

如果曾经把私钥上传到公共仓库，必须立即在 AWS IoT 里禁用/删除该证书并重新生成。

## 编译 ROS 2 工作区

```bash
cd ~/herbsecurity_ws/ros2_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

## 启动整套终端

```bash
cd ~/herbsecurity_ws/ros2_ws
source install/setup.bash
ros2 launch herb_security_ros herb_terminal.launch.py
```

## 单独测试称重

```bash
cd ~/herbsecurity_ws
python3 read_HX711.py
```

## 单独测试 K1 推理

```bash
cd ~/herbsecurity_ws
python3 run_gouqi_yolo_fp32_k1.py
```

## 上传 GitHub 的推荐方式

第一次上传：

```bash
cd herb-security-terminal
git init
git add .
git commit -m "init: add herb security terminal source code"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

后续更新：

```bash
git status
git add .
git commit -m "update: describe changes"
git push
```

## 不建议提交的内容

- 真实私钥、证书、token、密码。
- `build/`、`install/`、`log/`。
- `__pycache__/`、`*.pyc`。
- 超大模型文件，除非使用 Git LFS 或 GitHub Release。
- 个人测试数据、人脸库、数据库。
