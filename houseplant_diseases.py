import numpy as np
import cv2
import time
import onnxruntime as ort
#import spacemit_ort  # 进迭时空NPU加速库

import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # 保持长宽比缩放+补边，和YOLOv8一致
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(image, input_size=(640, 640)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 关键：替换直接resize，使用letterbox
    img, _, _ = letterbox(img, input_size)
    img = img / 255.0
    # HWC → CHW + batch维度
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def inference(session, img):
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: img}

    t0 = time.time()
    outputs = session.run(None, ort_inputs)
    t1 = time.time()

    print(f"推理耗时：{(t1-t0)*1000:.2f} ms")
    return outputs

CLASS_NAMES = ["yellowing", "rot", "mold", "wilt"]
CONF_THRES = 0.25
IOU_THRES = 0.5
INPUT_SIZE = (640, 640)


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

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union

        order = rest[iou <= iou_thres]

    return keep


def postprocess(outputs, orig_shape, conf_thres=0.25, iou_thres=0.5):
    pred = np.asarray(outputs[0])

    if pred.ndim == 3:
        pred = pred[0]

    # YOLOv8 常见输出为 [4 + num_classes, num_boxes]，需要转置
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4]          # xywh
    class_scores = pred[:, 4:]   # 各类别置信度

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    # 1. 置信度筛选 score > 0.25
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # 2. 坐标转换 xywh -> xyxy
    boxes = xywh2xyxy(boxes)

    # 3. NMS，按类别做偏移，避免不同类别互相抑制
    max_wh = 7680
    nms_boxes = boxes + class_ids[:, None] * max_wh
    keep = nms(nms_boxes, scores, iou_thres)

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # 4. 坐标映射回原图：去掉 letterbox padding，再除以缩放比例
    r, dw, dh = calc_letterbox_params(orig_shape, INPUT_SIZE)

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    detections = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        detections.append((box.astype(np.int32), float(score), int(cls_id)))

    return detections


def draw_detections(image, detections):
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
    ]

    for box, score, cls_id in detections:
        x1, y1, x2, y2 = box.tolist()
        color = colors[cls_id % len(colors)]

        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{class_name} {score:.2f}"

        # 画框 bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 写类别名 + 置信度
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y = max(y1, th + baseline + 4)

        cv2.rectangle(
            image,
            (x1, text_y - th - baseline - 4),
            (x1 + tw + 4, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return image

if __name__ == "__main__":
    # 1. 配置NPU加速会话
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2

    # 加载你转换好的jdsk模型
    model_path = "houseplant_diseases_int8.jdsk"
    session = ort.InferenceSession(
        model_path,
        sess_options=opt,
        providers=["SpaceMITExecutionProvider"]
    )

    # 2. 读取图片
    frame = cv2.imread("test.png")
    if frame is None:
        raise FileNotFoundError("test.jpg not found or failed to read")
    data = preprocess(frame)


    # 3. 运行推理
    result = inference(session, data)

    # 4. 后处理：置信度筛选、坐标转换、NMS、坐标映射、画框显示
    detections = postprocess(result, frame.shape, CONF_THRES, IOU_THRES)

    output = draw_detections(frame.copy(), detections)
    cv2.imwrite("result.png", output)

    print(f"检测到 {len(detections)} 个目标，结果已保存：result.png")