import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import cv2
import time

def get_image(path):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    return img

def preprocess(img):
    img = img / 255.0
    img = cv2.resize(img, (224, 224))
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def predict(path):
    img = get_image(path)
    img = preprocess(img)

    # 构造模型输入
    ort_inputs = {session.get_inputs()[0].name: img}

    start = time.time()
    preds = session.run(None, ort_inputs)[0]  # 执行推理
    end = time.time()

    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]  # 按置信度从高到低排序

    # 打印耗时 + 最优类别
    print('time=%.2fms; class=%s' % (round((end - start) * 1000, 2), label_dict[str(a[0])]))

with open("plant_to_name.json", encoding="utf-8") as f:
    label_dict = json.load(f)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2

# 加载量化后的INT8模型，启用K1硬件加速
session = ort.InferenceSession(
    'houseplant.jdsk',
    sess_options=session_options,
    providers=["SpacemitNPUExecutionProvider"]
)

img_path = './test.png' #要改成从摄像头采集
predict(img_path)