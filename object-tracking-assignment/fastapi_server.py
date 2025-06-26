from fastapi import FastAPI, WebSocket
# from track_high_range import track_data, country_balls_amount
# from track_high_amount import track_data, country_balls_amount
from track_high import track_data, country_balls_amount
import asyncio
import glob
import random
from PIL import Image
from io import BytesIO
import re
import base64
import os
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np
import supervision as sv
from collections import defaultdict
from datetime import datetime
import cv2
import json

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

tracker = sv.ByteTrack()

id_history = defaultdict(dict)

def detect_with_template_matching(frame, templates, threshold=0.8):
    detections = []
    for class_id, template in enumerate(templates):
        if template is None:
            continue

        if frame.shape[2] != template.shape[2]:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

        locations = np.where(result >= threshold)
        h, w, _ = template.shape

        rects = []
        for (x, y) in zip(*locations[::-1]):
            rects.append([int(x), int(y), int(x + w), int(y + h)])

        if not rects:
            continue
        
        boxes, _ = cv2.groupRectangles(rects, 1, 0.5)

        for (x1, y1, x2, y2) in boxes:
            confidence = np.mean(result[y1:y2, x1:x2])
            detections.append(([x1, y1, x2, y2], confidence, class_id))

    if not detections:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

    xyxy_boxes = np.array([d[0] for d in detections], dtype=np.float32)
    conf_scores = np.array([d[1] for d in detections], dtype=np.float32)
    class_ids = np.array([d[2] for d in detections])

    return xyxy_boxes, conf_scores, class_ids

def calculate_switches(current_frame_id, current_data):
    global id_history
    
    switches_this_frame = 0
    current_assignments = {}
    for obj in current_data:
        if obj.get('track_id') is not None:
            current_assignments[obj['cb_id']] = obj['track_id']
    
    if current_frame_id > 1:
        prev_frame_id = current_frame_id - 1
        if prev_frame_id in id_history:
            for cb_id, current_track_id in current_assignments.items():
                if cb_id in id_history[prev_frame_id]:
                    prev_track_id = id_history[prev_frame_id][cb_id]
                    if prev_track_id != current_track_id:
                        switches_this_frame += 1
    
    id_history[current_frame_id] = current_assignments
    return switches_this_frame

def tracker_soft(bbox2id, id2bbox, el, previous_bboxes=None):
    bbox_tmp = [x['bounding_box'] for x in el['data'] if x.get('bounding_box')]
    
    if not previous_bboxes or not bbox_tmp:
        for i, val in enumerate(el['data']):
            val['track_id'] = i
        switches = calculate_switches(el['frame_id'], el['data'])
        return bbox2id, id2bbox, el, bbox_tmp, switches

    IoU_matrix = box_iou(torch.Tensor(previous_bboxes), torch.Tensor(bbox_tmp))
    matrix = linear_sum_assignment(-IoU_matrix.numpy())
    
    current_assignments = {}
    for index in range(len(matrix[0])):
        index_bbox_old = matrix[0][index]
        index_bbox_new = matrix[1][index]
        old_bbox = tuple(previous_bboxes[index_bbox_old])
        new_bbox = tuple(bbox_tmp[index_bbox_new])
        if old_bbox in bbox2id:
            track_id = bbox2id.pop(old_bbox)
            current_assignments[new_bbox] = track_id

    next_id = max(id2bbox.keys(), default=-1) + 1
    for bbox in bbox_tmp:
        bbox_t = tuple(bbox)
        if bbox_t not in current_assignments:
            current_assignments[bbox_t] = next_id
            next_id += 1
    
    bbox2id = current_assignments
    id2bbox = {v: k for k, v in bbox2id.items()}

    for val in el['data']:
        bbox_val = tuple(val['bounding_box']) if val.get('bounding_box') else None
        val['track_id'] = bbox2id.get(bbox_val)

    switches = calculate_switches(el['frame_id'], el['data'])
    
    return bbox2id, id2bbox, el, bbox_tmp, switches

def tracker_strong(el, templates, frame_dir="save_frames_dir"):
    frame_id = el['frame_id']
    frame_path = os.path.join(frame_dir, f"frame_{frame_id - 1}.png")

    if not os.path.exists(frame_path):
        for det in el['data']: det['track_id'] = None
        return el, 0

    frame = cv2.imread(frame_path)
    if frame is None:
        for det in el['data']: det['track_id'] = None
        return el, 0

    xyxy, confs, class_ids = detect_with_template_matching(frame, templates)

    detection_obj = sv.Detections.empty()
    if xyxy.shape[0] > 0:
        detection_obj = sv.Detections(xyxy=xyxy, confidence=confs, class_id=class_ids)

    tracked_detections = tracker.update_with_detections(detection_obj)

    for item in el['data']: item['track_id'] = None

    valid_original_boxes = [d['bounding_box'] for d in el['data'] if d.get('bounding_box')]
    if len(tracked_detections) > 0 and len(valid_original_boxes) > 0:
        original_bboxes = torch.tensor(valid_original_boxes, dtype=torch.float)
        tracked_bboxes = torch.tensor(tracked_detections.xyxy, dtype=torch.float)
        iou_matrix = box_iou(original_bboxes, tracked_bboxes)
        orig_indices, track_indices = linear_sum_assignment(-iou_matrix.numpy())
        original_box_map = [i for i, d in enumerate(el['data']) if d.get('bounding_box')]
        for orig_idx, track_idx in zip(orig_indices, track_indices):
            if iou_matrix[orig_idx, track_idx] > 0.3:
                original_data_index = original_box_map[orig_idx]
                el['data'][original_data_index]['track_id'] = int(tracked_detections.tracker_id[track_idx])

    all_ids_are_none = all(item.get('track_id') is None for item in el['data'])
    if all_ids_are_none and el['data']:
        for i, item in enumerate(el['data']):
            item['track_id'] = i

    switches_this_frame = calculate_switches(el['frame_id'], el['data'])
    
    return el, switches_this_frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, USE_STRONG_TRACKER = True):
    print('Accepting client connection...')
    await websocket.accept()
    await websocket.send_text(str(country_balls))

    global id_history
    id_history = defaultdict(dict)
    
    total_switches = 0 
    frame_count = 0
    bbox_tmp = None
    previous_bboxes = None
    bbox2id = dict()
    id2bbox = dict()

    templates = [cv2.resize(cv2.imread(img_path), (50, 50)) for img_path in imgs if cv2.imread(img_path) is not None]
    if not templates:
        print("ОШИБКА: Не удалось загрузить ни одного шаблона из папки 'imgs/'.")
        await websocket.close()
        return

    for index, el in enumerate(track_data):
        await asyncio.sleep(0.5)
        frame_count += 1
        
        switches_this_frame = 0
        if USE_STRONG_TRACKER:
            processed_el, switches_this_frame = tracker_strong(el, templates=templates, frame_dir="save_frames_dir")
        else:
            bbox2id, id2bbox, processed_el, previous_bboxes, switches_this_frame = tracker_soft(
                bbox2id, id2bbox, el, previous_bboxes)
        
        total_switches += switches_this_frame
        await websocket.send_json(processed_el)
    
    avg_switches_per_frame = total_switches / frame_count if frame_count > 0 else 0
        
    metrics_data = f"""
------
Session: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Tracker type: {'strong' if USE_STRONG_TRACKER else 'soft'}
Total frames processed: {frame_count}
Total ID switches: {total_switches}
Average switches per frame: {avg_switches_per_frame:.2f}
------
"""
    with open('tracking_metrics.txt', 'a', encoding='utf-8') as f:
            f.write(metrics_data)
    print('Bye..')

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print('Accepting client connection...')
#     await websocket.accept()
#     await websocket.receive_text()

#     dir = "save_frames_dir"
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     await websocket.send_text(json.dumps(country_balls))
#     for el in track_data:
#         await asyncio.sleep(0.5)
#         image_data = await websocket.receive_text()
#         try:
#             image_data = re.sub('^data:image/.+;base64,', '', image_data)
#             image = Image.open(BytesIO(base64.b64decode(image_data)))
#             image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#             frame_id = el['frame_id'] - 1
#             image.save(f"{dir}/frame_{frame_id}.png")
#         except Exception as e:
#             print(e)
    
#         await websocket.send_json(el)

#     print('Bye..')