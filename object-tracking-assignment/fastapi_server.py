from fastapi import FastAPI, WebSocket
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

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

tracker = sv.ByteTrack()

id_switches = 0
id_history = defaultdict(dict)  

def calculate_switches(current_frame_id, current_data):
    global id_switches, id_history
    
    current_assignments = {}
    for obj in current_data:
        if obj['track_id'] is not None:
            current_assignments[obj['cb_id']] = obj['track_id']
    
    if current_frame_id > 1:
        prev_frame_id = current_frame_id - 1
        for cb_id, current_track_id in current_assignments.items():
            if cb_id in id_history[prev_frame_id]:
                prev_track_id = id_history[prev_frame_id][cb_id]
                if prev_track_id != current_track_id:
                    id_switches += 1
    
    id_history[current_frame_id] = current_assignments
    return id_switches

def tracker_soft(bbox2id, id2bbox, el, previous_bboxes = None):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    # random.shuffle(el['data'])
    bbox_tmp = [x['bounding_box'] for x in el['data'] if x['bounding_box']]

    if not previous_bboxes:
        for i, bbox in enumerate(bbox_tmp):
            bbox2id[tuple(bbox)] = i
            id2bbox[i] = bbox
        for val in el['data']:
            if val['bounding_box']:
                val['track_id'] = bbox2id[tuple(val['bounding_box'])]
        return bbox2id, id2bbox, el, bbox_tmp

    IoU_matrix = box_iou(
        torch.Tensor(previous_bboxes),
        torch.Tensor(bbox_tmp)
    )
    IoU_matrix = -IoU_matrix
    matrix = linear_sum_assignment(IoU_matrix)
    used_ids = set()

    for index in range(len(matrix[0])):
        index_bbox_old = matrix[0][index]
        index_bbox_new = matrix[1][index]
        old_bbox = tuple(previous_bboxes[index_bbox_old])
        new_bbox = tuple(bbox_tmp[index_bbox_new])
        if old_bbox in bbox2id:
            track = bbox2id.pop(old_bbox)
            id2bbox[track] = new_bbox
            bbox2id[new_bbox] = track
            used_ids.add(track)

    next_id = max(id2bbox.keys(), default=-1) + 1
    for bbox in bbox_tmp:
        bbox_t = tuple(bbox)
        if bbox_t not in bbox2id and len(bbox_t) > 0:
            bbox2id[bbox_t] = next_id
            id2bbox[next_id] = bbox
            next_id += 1

    for val in el['data']:
        bbox_val = tuple(val['bounding_box'])
        val['track_id'] = bbox2id.get(bbox_val, -1)
        # print(bbox2id.get(bbox_val, -1))
    current_switches = calculate_switches(el['frame_id'], el['data'])
    
    return bbox2id, id2bbox, el, bbox_tmp, current_switches

def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true и воспользуйтесь нижним закомментированным кодом в этом файле для первого прогона, 
    на повторном прогоне можете читать сохраненные фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    detections = el['data']
    
    xyxy = []
    confidences = []
    class_ids = []

    for det in detections:
        box = det['bounding_box']
        if box is None or len(box) == 0:
            x, y = det['x'], det['y']
            box = [x - 1, y - 1, x + 1, y + 1]
        xyxy.append(box)
        confidences.append(det.get('confidence', 1.0))
        class_ids.append(det['cb_id'])

    detection_obj = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids)
    )

    tracked_detections = tracker.update_with_detections(detection_obj)
    for det in detections:
        det['track_id'] = None
    for i in range(len(tracked_detections)):
        detections[i]['track_id'] = int(tracked_detections.tracker_id[i])

    current_switches = calculate_switches(el['frame_id'], el['data'])
        
    return el, current_switches


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    bbox_tmp = None
    bbox2id = dict()
    id2bbox = dict()
    for index, el in enumerate(track_data):
        await asyncio.sleep(0.5)
        # TODO: part 1
        # el = tracker_soft(el)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        # json_el = json.dumps(el)
        # print(el)
        # bbox2id, id2bbox, el, bbox_tmp = tracker_soft(bbox2id, id2bbox, el, bbox_tmp)
        # print(el)
        # await websocket.send_json(el)
        # if index == 2:
        #     break
        # print(el)
        if USE_STRONG_TRACKER:
            processed_el, switches = tracker_strong(el)
        else:
            bbox2id, id2bbox, processed_el, previous_bboxes, switches = tracker_soft(
                bbox2id, id2bbox, el, previous_bboxes)
        
        print(f"Frame {el['frame_id']} - ID switches: {switches}")
        await websocket.send_json(processed_el)
    
    print(f'Total ID switches: {id_switches}')
    print('Bye..')
    # добавьте сюда код рассчета метрики
    bbox_tmp = None


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print('Accepting client connection...')
#     await websocket.accept()
#     await websocket.receive_text()
#     # отправка служебной информации для инициализации объектов
#     # класса CountryBall на фронте

#     dir = "save_frames_dir"
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     await websocket.send_text(str(country_balls))
#     for el in track_data:
#         await asyncio.sleep(0.5)
#         image_data = await websocket.receive_text()
#         # print(image_data)
#         try:
#             image_data = re.sub('^data:image/.+;base64,', '', image_data)
#             image = Image.open(BytesIO(base64.b64decode(image_data)))
#             image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#             frame_id = el['frame_id'] - 1
#             image.save(f"{dir}/frame_{frame_id}.png")
#             # print(image)
#         except Exception as e:
#             print(e)
    
#         # отправка информации по фрейму
#         await websocket.send_json(el)

#     await websocket.send_json(el)
#     await asyncio.sleep(0.5)
#     image_data = await websocket.receive_text()
#     try:
#         image_data = re.sub('^data:image/.+;base64,', '', image_data)
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         image = image.resize((1000, 800), Image.Resampling.LANCZOS)
#         image.save(f"{dir}/frame_{el['frame_id']}.png")
#     except Exception as e:
#         print(e)

#     print('Bye..')
