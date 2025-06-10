1) Данные размечены в cvat, разбиение по кадрам осуществляется с помощью cv2
2) При подготовке данных изображения переводится в формат матриц 
image = Image.open(self.path_im / Path(image_path)).convert('RGB')
im_arr = np.asarray(image)

im_arr = self.trasnform(image = im_arr)['image']

формирование label: перерасчет координат для сетки модели, извлечение корня из w, h, классы и confidence в строках тензора с объектам просто заполняются 1
for line in file.readlines():
  parts = [float(l) for l in line.split()]
                
                class_id = int(parts[0])
                x_center_norm = parts[1]
                y_center_norm = parts[2]
                width_norm = parts[3]
                height_norm = parts[4]

                x_grid = min(int(x_center_norm * self.S), self.S - 1)
                y_grid = min(int(y_center_norm * self.S), self.S - 1)

                x_cell = (x_center_norm * self.S) - x_grid
                y_cell = (y_center_norm * self.S) - y_grid
                
                w_sqrt = math.sqrt(width_norm)
                h_sqrt = math.sqrt(height_norm)

                class_one_hot = torch.zeros(self.C)
                class_one_hot[class_id] = 1.0
                
                box_info = torch.tensor([1.0, x_cell, y_cell, w_sqrt, h_sqrt], dtype=torch.float32)
                
                labels_tensor[y_grid, x_grid, :5] = box_info 
                labels_tensor[y_grid, x_grid, 5 : 5 + self.C] = class_one_hot

3) реализован перерасчет координт, который используется в дальнейшем в nms и расчете лосса модели, по сути просто получение x_min, x_max, y_min, y_max, сама функция перерасчета def calc_coords(inp):

4)реализован расчет функции потерь модели: на первых шагах происходит формирование mask для извлечения соответствующих строк, которые содержат объекты из тензора mask = label[:, :, :, 0] > 0, а также маска для извлечения строк не содержащих объектов mask_no = label[:, :, :, 0] == 0, данная маска будет применяться для получения матрицы не содержащая объектов для расчета conf, в цикле ищем среди боксов те которые имеют наибольшее значение по iou, перед расчетом переводим координаты с помощью calc_coords метода, для использования box_iou, получаем диагональные значения, 

iou_matrix = box_iou(real_coords_p, real_coords_l)
iou_values = torch.diag(iou_matrix)
iou_res[:, index] = iou_values

из этих диагональных значений формируем iou_res[:, index] = iou_values матрицу, котороя будет содержать в себе значения box_iou, далее находим координаты в матрице тех bbox, предсказанных моделью, которые имеют наибольшее значение по box_iou с истинным bbox (coord_b = iou_res.argmax(dim=1)), полученные значения можно использовать как индексы B, для формирования матрицы, с векторами координат bbox, для этого сначала номер B нужно перевести в стартовую координату внутри матрицы pred, также и с conf

most_sim_start = (coord_b * 5 + 1).to(obj_p.device)
conf_ind = (coord_b * 5).to(obj_p.device)

после этого формирую векто, который содержит инжексы bbox и с помощью torch.gather извлекаю эти значения bbox

bbox_ind = most_sim_start.unsqueeze(1) + torch.arange(4, device = obj_p.device).unsqueeze(0) 
bbox_pred = torch.gather(input = obj_p, index = bbox_ind, dim = 1)

далее после того как я извлек все необходимые матрицы идет просто расчет каждого из лоссов

5) Добавлены метрики для проверки модели на валидации, в которых также происходит извлечение необходимых значений, пример в методе validation_step
6) Реализован метод nms, иттерируясь по изображениям в батче получаем val_bboxes_only = image_preds[..., :B*5], то есть координаты bbox, применяю reshape для упрощения вида тензора и получения матрицы, после перевожу conf в уверенности используя sigmoid, на основании этого произвожу сортировку значения по conf, sort_indices=torch.sort(tensor_single_bbox[:, 0], descending=True), чтобы в дальнейшем отсеить те bbox, которые имеют conf ниже установленного порога mask_conf = sorted_tensor[:, 0] > conf_thresh, следующим шагом в цикле пока не закончатся bbox, добавляю наибольший по conf bbox в лист, считаю box_iou для текущего со всеми, получаю матрицу, которая содержит iou, ее же буду использовать как маску для отсеивания тех боксов, которые имеют наибольший iou с текущим bbox.

7) Реализована метрика расчета в данной функции мы извлекаем изображения из батча и получаем строки в которых есть объект, далее получаем bbox и классы объектов из таргета, после мы добавляем эти значения в словарь, после делаем то же самое для pred, в них с помощью nms получаем боксы, в которых наибольший conf, собираем истинные метки классов и предсказания модели по меткам классов, после этого собираем TP и FP 1, если iou больше threshold и метки классов совпадают потом находим среднее TP и FP.

 

![Image](https://github.com/user-attachments/assets/d3fefea7-7eb0-4a2b-95db-b06c3481dc1e)
https://github.com/1MaxOn1/CV_homeworks/issues/2
