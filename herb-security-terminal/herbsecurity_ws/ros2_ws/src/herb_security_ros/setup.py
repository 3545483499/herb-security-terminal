from setuptools import setup
from glob import glob

package_name = 'herb_security_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name + '/ui', glob('ui/*.ui')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='mjn',
    maintainer_email='mjn@example.com',
    description='MUSE Pi Pro herb security terminal',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MainWindow = herb_security_ros.MainWindow:main',
            'login_auth_node = herb_security_ros.login_auth_node:main',
            'aht30_node = herb_security_ros.aht30_node:main',
            'herb_recognition_node = herb_security_ros.herb_recognition_node:main',

            # 保留入口，但 launch 不启动也没事
            'aws_iot_bridge = herb_security_ros.aws_iot_bridge:main',
        ],
    },
)