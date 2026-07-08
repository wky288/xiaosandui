#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能花盆系统 - 一体化服务
包含：摄像头采集 + AI推理 + HTTP API + 图片Base64
"""

import os
import sys
import time
import json
import math
import base64
import subprocess
import threading
import cv2
import numpy as np
import onnxruntime as ort
import requests
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

# ============================================================
# Flask API 服务
# ============================================================
app = Flask(__name__)
CORS(app)

# 全局数据缓存
latest_data = {
    "plant_name": "等待采集...",
    "disease": "无",
    "confidence": 0,
    "advice": "等待首次采集...",
    "sensors": {
        "temperature": 0,
        "humidity": 0,
        "light": 0
    },
    "weather": "获取中...",
    "timestamp": "",
    "image_path": "/tmp/plant_photo.jpg",
    "image_base64": ""
}

start_time = datetime.now()

def update_data(plant_name, disease, confidence, advice, temp, hum, light, weather):
    """更新缓存数据"""
    global latest_data
    latest_data = {
        "plant_name": plant_name,
        "disease": disease,
        "confidence": confidence,
        "advice": advice,
        "sensors": {
            "temperature": temp,
            "humidity": hum,
            "light": light
        },
        "weather": weather,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": "/tmp/plant_photo.jpg",
        "image_base64": get_image_base64()
    }
    print(f"[API] 数据已更新: {plant_name}, {disease}")

def get_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

def get_image_base64():
    """读取图片并转换为Base64"""
    if os.path.exists("/tmp/plant_photo.jpg"):
        try:
            with open("/tmp/plant_photo.jpg", "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except:
            return ""
    return ""

# ============================================================
# API 路由
# ============================================================
@app.route('/')
def index():
    return jsonify({
        "name": "智能花盆系统 API",
        "version": "1.0",
        "endpoints": {
            "/api/latest": "获取最新数据（含图片Base64）",
            "/api/status": "获取设备状态",
            "/image": "获取最新照片（直接下载）",
            "/api/update": "手动触发采集（POST）"
        }
    })

@app.route('/api/latest')
def get_latest():
    """返回最新数据，包含图片Base64"""
    data = latest_data.copy()
    # 确保 image_base64 字段存在
    if "image_base64" not in data or not data["image_base64"]:
        data["image_base64"] = get_image_base64()
    return jsonify(data)

@app.route('/api/status')
def get_status():
    uptime = str(datetime.now() - start_time).split('.')[0]
    return jsonify({
        "connected": True,
        "ip": get_ip(),
        "uptime": uptime,
        "camera_status": "OK" if os.path.exists("/tmp/plant_photo.jpg") else "No image",
        "sensor_status": "OK",
        "model_status": "OK",
        "last_update": latest_data.get("timestamp", "")
    })

@app.route('/image')
def get_image():
    if os.path.exists("/tmp/plant_photo.jpg"):
        return send_file("/tmp/plant_photo.jpg", mimetype='image/jpeg')
    return jsonify({"error": "图片不存在"}), 404

@app.route('/api/update', methods=['POST'])
def trigger_update():
    """手动触发一次采集"""
    return jsonify({"status": "采集已触发", "note": "请稍后查询 /api/latest"})

# ============================================================
# 启动 API 服务的函数
# ============================================================
def start_api():
    """在单独线程中启动 Flask API"""
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)

# ============================================================
# 配置
# ============================================================
MODEL_PATH_YOLO = "/home/bianbu/models/houseplant_diseases.jdsk"
MODEL_PATH_RESNET = "/home/bianbu/models/houseplant_int8.jdsk"

YOLO_CLASS_NAMES = ["Leaf yellowing", "Rot", "Mold", "Wilt"]
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.5
YOLO_INPUT_SIZE = (640, 640)

RESNET_INPUT_SIZE = (224, 224)

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

WEATHER_KEY = "SQZ17VOeM3YvE5AvM"
WEATHER_LOCATION = "xiamen"

ADC_BASE = "/sys/bus/iio/devices/iio:device0"
ADC_TEMP = f"{ADC_BASE}/in_voltage0_raw"
ADC_SOIL = f"{ADC_BASE}/in_voltage1_raw"
ADC_LIGHT = f"{ADC_BASE}/in_voltage2_raw"

PHOTO_INTERVAL = 30
FRAME_PATH = "/tmp/cpp0_output_1920x1080_s1920.nv12"
OUTPUT_JPG = "/tmp/plant_photo.jpg"

# 植物名称映射
PLANT_NAME_MAP = {
    "0": "秋海棠", "1": "镜面草", "2": "非洲紫罗兰", "3": "芦荟",
    "4": "红掌", "5": "散尾葵", "6": "鸟巢蕨", "7": "波士顿蕨",
    "8": "竹芋", "9": "康乃馨", "10": "一叶兰", "11": "广东万年青",
    "12": "蟹爪兰", "13": "菊花", "14": "雏菊", "15": "水仙",
    "16": "龙血树", "17": "花叶万年青", "18": "海芋", "19": "常春藤",
    "20": "风信子", "21": "铁十字秋海棠", "22": "玉树", "23": "铃兰",
    "24": "发财树", "25": "兰花", "26": "袖珍椰子", "27": "白掌",
    "28": "一品红", "29": "酒瓶兰", "30": "绿萝", "31": "祈祷草",
    "32": "响尾蛇竹芋", "33": "月季", "34": "橡皮树", "35": "鹅掌柴",
    "36": "虎皮兰", "37": "向日葵", "38": "龟背竹", "39": "紫露草",
    "40": "郁金香", "41": "睡莲"
}

# ============================================================
# 显示环境检测
# ============================================================
if "DISPLAY" not in os.environ:
    SHOW_WINDOW = False
    print("[INFO] 无图形显示环境，将不显示窗口，结果保存到 photos/ 目录")
else:
    SHOW_WINDOW = True
    print("[INFO] 图形显示环境可用")

# ============================================================
# 加载中文字体
# ============================================================
def get_chinese_font(size=16):
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/home/bianbu/.fonts/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

# ============================================================
# 加载模型
# ============================================================
def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        sys.exit(1)
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=opt,
            providers=["CPUExecutionProvider"]
        )
        print(f"[INFO] 模型加载成功: {model_path}")
        return session
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        sys.exit(1)

# ============================================================
# 拍照模块
# ============================================================
def capture_photo():
    print(f"[{time.strftime('%H:%M:%S')}] 正在拍照...", end="", flush=True)

    if os.path.exists(FRAME_PATH):
        os.remove(FRAME_PATH)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "wayland"
    env["DISPLAY"] = ":0"

    try:
        subprocess.run(
            ["cam-test", "/tmp/csi3_camera_auto.json"],
            env=env,
            timeout=10
        )
    except:
        pass

    if os.path.exists(FRAME_PATH):
        with open(FRAME_PATH, 'rb') as f:
            raw_data = f.read()
        expected_size = 1920 * 1080 * 3 // 2
        if len(raw_data) >= expected_size:
            try:
                subprocess.run([
                    "ffmpeg", "-f", "rawvideo", "-pix_fmt", "nv12",
                    "-s", "1920x1080",
                    "-i", FRAME_PATH,
                    "-frames:v", "1",
                    OUTPUT_JPG, "-y"
                ], capture_output=True, check=True)
                print(" 成功")
                return cv2.imread(OUTPUT_JPG)
            except Exception as e:
                print(f" 转换失败: {e}")
                return None

    print(" 失败")
    return None

# ============================================================
# YOLO 预处理 + 后处理
# ============================================================
def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, r, (dw, dh)

def preprocess_yolo(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img, _, _ = letterbox(img, YOLO_INPUT_SIZE)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def nms(boxes, scores, iou_thres=0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thres]
    return keep

def xywh2xyxy(boxes):
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy

def calc_letterbox_params(orig_shape, input_size=(640, 640)):
    h, w = orig_shape[:2]
    r = min(input_size[0] / h, input_size[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (input_size[1] - new_w) / 2
    dh = (input_size[0] - new_h) / 2
    return r, dw, dh

def postprocess_yolo(outputs, orig_shape):
    pred = np.asarray(outputs[0])
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores > YOLO_CONF_THRES
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes = xywh2xyxy(boxes)

    max_wh = 7680
    nms_boxes = boxes + class_ids[:, None] * max_wh
    keep = nms(nms_boxes, scores, YOLO_IOU_THRES)

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    r, dw, dh = calc_letterbox_params(orig_shape, YOLO_INPUT_SIZE)
    h, w = orig_shape[:2]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    detections = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        detections.append((box.astype(np.int32), float(score), int(cls_id)))
    return detections

def draw_detections(image, detections):
    colors = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255)]
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = box.tolist()
        color = colors[cls_id % len(colors)]
        class_name = YOLO_CLASS_NAMES[cls_id] if cls_id < len(YOLO_CLASS_NAMES) else str(cls_id)
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(y1, th + baseline + 4)
        cv2.rectangle(image, (x1, text_y - th - baseline - 4), (x1 + tw + 4, text_y + baseline), color, -1)
        cv2.putText(image, label, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image

# ============================================================
# ResNet 预处理
# ============================================================
def preprocess_resnet(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = cv2.resize(img, RESNET_INPUT_SIZE)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ============================================================
# 传感器读取
# ============================================================
def read_adc(path):
    try:
        with open(path, 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def read_sensors():
    adc0 = read_adc(ADC_TEMP)
    adc1 = read_adc(ADC_SOIL)
    adc2 = read_adc(ADC_LIGHT)

    if adc0 > 0:
        R = 10000.0 * (4095.0 - adc0) / adc0
        temp = 1.0 / (1.0/298.15 + math.log(R/10000.0)/3950.0) - 273.15
    else:
        temp = 25.0

    hum = 100 - (adc1 - 800) * 100 / (4095 - 800)
    hum = max(0, min(100, hum))

    light = (4095 - adc2) / 4095.0 * 1000
    light = max(0, min(1000, light))

    return temp, hum, light

# ============================================================
# 天气预报
# ============================================================
def get_weather():
    try:
        url = "https://api.seniverse.com/v3/weather/now.json"
        params = {"key": WEATHER_KEY, "location": WEATHER_LOCATION, "language": "zh-Hans", "unit": "c"}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if "results" in data:
            now = data["results"][0]["now"]
            return f"{now['text']}，{now['temperature']}°C"
    except:
        pass
    return "获取失败"

# ============================================================
# 大模型调用
# ============================================================
def ask_ollama(prompt):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": 80}
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except:
        pass
    return "大模型服务不可用"

# ============================================================
# PIL 中文绘制工具
# ============================================================
def cv2_to_pil(cv2_img):
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    rgb_img = np.array(pil_img)
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

def draw_chinese_text_pil(cv2_img, text, position, font_size=16, color=(255, 255, 255)):
    pil_img = cv2_to_pil(cv2_img)
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(font_size)
    draw.text(position, text, font=font, fill=color)
    return pil_to_cv2(pil_img)

def draw_chinese_text_multiline(cv2_img, text, position, font_size=16, color=(255, 255, 255), line_spacing=5):
    x, y = position
    words_per_line = 14
    lines = []
    current_line = ""
    for char in text:
        if len(current_line) < words_per_line:
            current_line += char
        else:
            lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)

    result = cv2_img.copy()
    for i, line in enumerate(lines):
        result = draw_chinese_text_pil(result, line, (x, y + i * (font_size + line_spacing)), font_size, color)
    return result

# ============================================================
# 仪表板绘制
# ============================================================
def draw_dashboard_800x480(left_img, plant_name, disease, disease_conf, temp, hum, light, weather, advice):
    display_height = 480
    panel_width = 280

    h, w = left_img.shape[:2]
    scale = display_height / h
    new_w = int(w * scale)
    left_display = cv2.resize(left_img, (new_w, display_height))

    if new_w > 520:
        left_display = left_display[:, :520]

    right_panel = np.ones((display_height, panel_width, 3), dtype=np.uint8) * 30

    right_panel = draw_chinese_text_pil(right_panel, "植物监测", (10, 8), font_size=14, color=(255, 255, 255))
    cv2.line(right_panel, (10, 28), (panel_width - 10, 28), (80, 80, 80), 1)

    y = 42
    lh = 20

    right_panel = draw_chinese_text_pil(right_panel, f"温度 {temp:.0f}C  湿度 {hum:.0f}%  光照 {light:.0f}lx", (10, y), font_size=11, color=(255, 255, 255))
    y += lh

    right_panel = draw_chinese_text_pil(right_panel, f"天气 {weather}", (10, y), font_size=11, color=(255, 255, 255))
    y += lh + 2

    right_panel = draw_chinese_text_pil(right_panel, f"植物 {plant_name}", (10, y), font_size=13, color=(0, 255, 255))
    y += lh + 2

    if disease == "无病害":
        right_panel = draw_chinese_text_pil(right_panel, f"健康", (10, y), font_size=12, color=(0, 255, 0))
    else:
        right_panel = draw_chinese_text_pil(right_panel, f"{disease} ({disease_conf:.0%})", (10, y), font_size=12, color=(0, 0, 255))
    y += lh + 2

    right_panel = draw_chinese_text_pil(right_panel, "建议:", (10, y), font_size=12, color=(0, 255, 255))
    y += lh - 2

    if len(advice) > 20:
        part1 = advice[:20]
        part2 = advice[20:40]
        part3 = advice[40:60]
        right_panel = draw_chinese_text_pil(right_panel, part1, (10, y), font_size=10, color=(0, 255, 0))
        y += lh - 4
        if part2:
            right_panel = draw_chinese_text_pil(right_panel, part2, (10, y), font_size=10, color=(0, 255, 0))
            y += lh - 4
        if part3:
            right_panel = draw_chinese_text_pil(right_panel, part3, (10, y), font_size=10, color=(0, 255, 0))
    else:
        right_panel = draw_chinese_text_pil(right_panel, advice, (10, y), font_size=11, color=(0, 255, 0))

    right_panel = draw_chinese_text_pil(right_panel, "[q]退出", (10, display_height - 15), font_size=10, color=(128, 128, 128))

    dashboard = np.hstack([left_display, right_panel])
    if dashboard.shape[1] > 800:
        dashboard = dashboard[:, :800]
    return dashboard

# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("🌱 智能花盆系统 v5.0 (一体化服务 + 图片Base64)")
    print(f"   IP 地址: {get_ip()}")
    print("   API 端口: 5002")
    print(f"   拍照间隔: {PHOTO_INTERVAL} 秒")
    print("=" * 60)

    # 启动 API 服务（在后台线程）
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    print("[INFO] HTTP API 服务已启动 (端口 5002)")

    try:
        font = get_chinese_font(16)
        print("[INFO] 中文字体加载成功")
    except:
        print("[WARN] 中文字体加载失败")

    if not SHOW_WINDOW:
        os.makedirs("/home/bianbu/photos", exist_ok=True)

    print("[INFO] 加载模型中...")
    session_yolo = load_model(MODEL_PATH_YOLO)
    session_resnet = load_model(MODEL_PATH_RESNET)
    print("[INFO] 所有模型加载完成")

    print("[INFO] 系统启动成功！按 'q' 退出")
    print("=" * 60)

    count = 0
    try:
        while True:
            count += 1
            start_time_loop = time.time()
            print(f"\n[第 {count} 次采集]")

            img = capture_photo()
            if img is None:
                print("[WARN] 拍照失败，等待重试...")
                time.sleep(5)
                continue

            img_yolo = preprocess_yolo(img)
            yolo_out = session_yolo.run(None, {session_yolo.get_inputs()[0].name: img_yolo})
            detections = postprocess_yolo(yolo_out, img.shape)

            disease = "无病害"
            disease_conf = 0.0
            if detections:
                best = max(detections, key=lambda x: x[1])
                _, disease_conf, cls_id = best
                disease = YOLO_CLASS_NAMES[cls_id]

            img_resnet = preprocess_resnet(img)
            resnet_out = session_resnet.run(None, {session_resnet.get_inputs()[0].name: img_resnet})
            pred_id = str(np.argmax(resnet_out[0]))
            plant_name = PLANT_NAME_MAP.get(pred_id, f"未知({pred_id})")

            temp, hum, light = read_sensors()
            weather = get_weather()

            img_with_box = img.copy()
            if detections:
                img_with_box = draw_detections(img_with_box, detections)

            # 生成建议
            advice = "正在获取建议..."
            if disease != "无病害" and disease_conf > 0.3:
                prompt = f"""植物: {plant_name}, 病害: {disease}, 置信度: {disease_conf:.0%}, 温度: {temp:.1f}C, 湿度: {hum:.0f}%, 光照: {light:.0f}lx, 天气: {weather}
请给出简短养护建议（20字内）："""
            else:
                prompt = f"""植物: {plant_name}, 健康, 温度: {temp:.1f}C, 湿度: {hum:.0f}%, 光照: {light:.0f}lx, 天气: {weather}
请给出日常养护建议（20字内）："""
            advice = ask_ollama(prompt)

            # ============================================================
            # 同步数据到 API（关键！）
            # ============================================================
            try:
                update_data(
                    plant_name=plant_name,
                    disease=disease,
                    confidence=int(disease_conf * 100),
                    advice=advice,
                    temp=temp,
                    hum=hum,
                    light=light,
                    weather=weather
                )
                print("   [API] 数据已同步")
            except Exception as e:
                print(f"   [WARN] API 数据同步失败: {e}")

            print(f"   植物: {plant_name}")
            print(f"   病害: {disease} (置信度: {disease_conf:.0%})")
            print(f"   温度: {temp:.1f}°C  湿度: {hum:.0f}%  光照: {light:.0f}lx")
            print(f"   天气: {weather}")
            print(f"   💡 {advice}")

            dashboard = draw_dashboard_800x480(
                img_with_box, plant_name, disease, disease_conf,
                temp, hum, light, weather, advice
            )

            if SHOW_WINDOW:
                cv2.imshow("Smart Planter", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 用户退出")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"/home/bianbu/photos/screenshot_{timestamp}.jpg"
                    cv2.imwrite(save_path, dashboard)
                    print(f"   📷 截图已保存: {save_path}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"/home/bianbu/photos/result_{timestamp}.jpg"
                cv2.imwrite(save_path, dashboard)
                print(f"   📷 结果已保存: {save_path}")

            elapsed = time.time() - start_time_loop
            wait = max(0, PHOTO_INTERVAL - elapsed)
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[INFO] 用户退出")
    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()