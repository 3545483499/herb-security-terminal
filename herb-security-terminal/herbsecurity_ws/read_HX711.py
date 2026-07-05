#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import statistics
import gpiod

CHIP_NAME = "gpiochip0"

# 你的实际接线
SCK_LINE = 49       # HX711 SCK  -> GPIO_49_3V3
DOUT_LINE = 50      # HX711 DOUT -> GPIO_50_3V3

# 1 个额外脉冲：A 通道，128 倍增益
GAIN_PULSES = 1


def to_signed_24(value):
    if value & 0x800000:
        value -= 0x1000000
    return value


def read_once(dout, sck):
    """
    读取 HX711 一次 24bit 原始值。
    注意：这里不加 sleep，尽量减小 SCK 高电平时间。
    """

    timeout = time.time() + 1.0

    # DOUT 拉低表示数据准备好
    while dout.get_value() == 1:
        if time.time() > timeout:
            raise TimeoutError("DOUT 一直为高，HX711 未准备好")
        time.sleep(0.001)

    value = 0

    for _ in range(24):
        sck.set_value(1)
        bit = dout.get_value()
        sck.set_value(0)

        value = (value << 1) | bit

    # 第 25 个脉冲，设置下一次为 A 通道 128 倍增益
    for _ in range(GAIN_PULSES):
        sck.set_value(1)
        sck.set_value(0)

    return to_signed_24(value)


def read_median(dout, sck, times=15):
    """
    读取多次，取中位数，滤掉偶发毛刺。
    """
    values = []

    for _ in range(times):
        try:
            raw = read_once(dout, sck)

            # 过滤明显异常值
            # 你刚才出现过 0x0FFFFF 这种跳变值，所以这里做简单保护
            if raw != -1 and raw != 0 and abs(raw) < 8000000:
                values.append(raw)

        except Exception:
            pass

    if len(values) < 3:
        raise RuntimeError("有效采样太少，请检查 HX711 接线或供电")

    return int(statistics.median(values))


def tare(dout, sck, times=30):
    """
    空秤去皮。
    """
    print("请保持称重传感器上没有物体，正在去皮...")
    time.sleep(1)

    values = []

    for _ in range(times):
        values.append(read_median(dout, sck, times=7))
        time.sleep(0.05)

    offset = int(statistics.median(values))
    print(f"去皮完成，OFFSET = {offset}")

    return offset


def calibrate(dout, sck, offset, known_weight_g):
    """
    用已知砝码校准比例系数。
    """
    print()
    print(f"请放上 {known_weight_g} g 的已知重量物体。")
    input("放稳后按 Enter 开始校准...")

    print("正在采样，请不要碰传感器...")
    time.sleep(1)

    values = []

    for _ in range(30):
        raw = read_median(dout, sck, times=7)
        values.append(raw)
        time.sleep(0.05)

    raw_known = int(statistics.median(values))
    delta = raw_known - offset

    if delta == 0:
        raise RuntimeError("校准失败：放上砝码后 raw 没有变化")

    scale = delta / known_weight_g

    print(f"校准完成")
    print(f"已知重量 raw = {raw_known}")
    print(f"delta = {delta}")
    print(f"SCALE = {scale:.6f} counts/g")

    return scale


def main():
    parser = argparse.ArgumentParser(description="HX711 weight reader for MUSE Pi Pro")
    parser.add_argument(
        "--cal",
        type=float,
        default=0,
        help="校准重量，单位 g。例如 --cal 100 表示用 100g 物体校准"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0,
        help="已经校准好的 SCALE，单位 counts/g"
    )
    args = parser.parse_args()

    print("HX711 称重读取程序")
    print(f"SCK  : {CHIP_NAME} line {SCK_LINE}")
    print(f"DOUT : {CHIP_NAME} line {DOUT_LINE}")
    print()

    chip = gpiod.Chip(CHIP_NAME)
    sck = chip.get_line(SCK_LINE)
    dout = chip.get_line(DOUT_LINE)

    sck.request(
        consumer="hx711_sck",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[0]
    )

    dout.request(
        consumer="hx711_dout",
        type=gpiod.LINE_REQ_DIR_IN
    )

    try:
        # 保证 SCK 初始为低电平
        sck.set_value(0)
        time.sleep(0.5)

        offset = tare(dout, sck)

        if args.cal > 0:
            scale = calibrate(dout, sck, offset, args.cal)
        elif args.scale != 0:
            scale = args.scale
            print(f"使用手动输入 SCALE = {scale:.6f} counts/g")
        else:
            scale = 1.0
            print("当前未校准，weight 显示的是相对原始值，不是真实克数。")
            print("建议使用：sudo /usr/bin/python3 hx711_weight.py --cal 100")
            print()

        print()
        print("开始连续读取，按 Ctrl+C 退出")
        print("raw：原始值；delta：去皮后变化量；weight：重量 g")
        print()

        while True:
            raw = read_median(dout, sck, times=15)
            delta = raw - offset
            weight = delta / scale

            # 小范围零漂抑制
            if abs(weight) < 0.3:
                weight = 0.0

            now = time.strftime("%H:%M:%S")

            print(
                f"[{now}] "
                f"raw={raw:>10d}    "
                f"delta={delta:>10d}    "
                f"weight={weight:>8.2f} g"
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n程序退出")

    finally:
        try:
            sck.set_value(0)
            sck.release()
            dout.release()
            chip.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()