from pathlib import Path

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_name = "herb_security_ros"
    package_share = Path(get_package_share_directory(package_name))

    login_auth_config = package_share / "config" / "login_auth.yaml"
    aht30_config = package_share / "config" / "aht30_node.yaml"
    herb_recognition_config = package_share / "config" / "herb_recognition.yaml"

    login_auth_node = Node(
        package=package_name,
        executable="login_auth_node",
        name="login_auth_node",
        output="screen",
        parameters=[str(login_auth_config)],
    )

    aht30_node = Node(
        package=package_name,
        executable="aht30_node",
        name="aht30_node",
        output="screen",
        parameters=[str(aht30_config)],
    )

    # 节点启动但默认 standby；只有进入出入库界面后 MainWindow 发 start 才会打开摄像头并推理。
    herb_recognition_node = Node(
        package=package_name,
        executable="herb_recognition_node",
        name="herb_recognition_node",
        output="screen",
        parameters=[str(herb_recognition_config)],
    )

    main_window_node = Node(
        package=package_name,
        executable="MainWindow",
        name="main_window_ui_node",
        output="screen",
    )

    return LaunchDescription([
        SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
        login_auth_node,
        aht30_node,
        herb_recognition_node,
        main_window_node,
    ])
