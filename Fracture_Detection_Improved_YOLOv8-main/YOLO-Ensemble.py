model_paths = ['GAM_best.pt',
               'SA_best.pt',
               'resblock_best.pt',
               'ECA_best.pt',
               'yolov9_best.pt',
               'yolov10_best.pt'
              ]
models = [YOLO(f'models/{model_path}') for model_path in model_paths]

images_dir = 'data/images'
labels_dir = 'data/labels'

df = pd.DataFrame(columns=['image_id', 'image_name'] + 
                  [f'{model[:-8]}_model_boxes' for model in model_paths] + 
                  [
                   'average_ensemble', 
                   # 'max_confidence_ensemble',
                   # 'nms_ensemble',
                    'true_boxes'
                  ])

def get_predictions(model, image_path):
    results = model(image_path)
    boxes = results[0].boxes.xywhn.cpu().numpy()  # Extract bounding boxes and move to CPU if using GPU
    return boxes.tolist() if len(boxes) > 0 else []

def load_ground_truth(label_path):
    with open(label_path, 'r') as file:
        boxes = []
        for line in file:
            parts = line.strip().split()
            # class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append([
                # class_id, 
                x_center, y_center, width, height])
    return boxes if len(boxes) > 0 else []

from collections import Counter
def get_voted_length(predictions_list):
    lengths = [len(prediction) for prediction in predictions_list]
    freq = Counter(lengths)
    print(freq)
    voted_len = test_freq.most_common()[0][0]
    return voted_len if voted_len != 0 else max(freq)

def get_average_boxes(length, predictions_list):
    average_boxes = [[0] * 4] * length
    last = []
    print(length)
    for i in range(length):
        involved = 0
        temp_box = [0] * 4
        for predictions in predictions_list:
            predictions_len = len(predictions)
            if predictions_len > i:
                involved += 1
                for j in range(4):
                    temp_box[j] += predictions[i][j]

        average_boxes[i] = [num / involved for num in temp_box]

    return average_boxes
            
    

def average_ensemble(predictions_list):
    if not any(predictions_list):
        return []
    else:
        voted_len = get_voted_length(predictions_list)
        
    if voted_len == 0:
        return []
        
    average_boxes = get_average_boxes(voted_len, predictions_list)
    return average_boxes

def max_confidence_ensemble(predictions_list):
    if not any(predictions_list):
        return None
    max_conf_boxes = max(predictions_list, key=lambda x: x[4] if x is not None else -1)
    return max_conf_boxes

def nms_ensemble(predictions_list, iou_threshold=0.5):
    from torchvision.ops import nms
    boxes = [box for sublist in predictions_list for box in sublist if sublist is not None]
    if len(boxes) == 0:
        return None
    boxes = np.array(boxes)
    scores = boxes[:, 4]
    boxes_tensor = torch.tensor(boxes[:, :4])
    keep = nms(boxes_tensor, torch.tensor(scores), iou_threshold)
    nms_boxes = boxes[keep]
    return nms_boxes.tolist()

image_id = 1
for image_name in os.listdir(images_dir):
    if image_name.endswith(('.jpg', '.png')):  # Modify as per your image formats
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        row = {'image_id': image_id, 'image_name': image_name}
        
        # Get model predictions
        predictions_list = []
        for model_name, model in zip(model_paths, models):
            predictions = get_predictions(model, image_path)
            row[f'{model_name[:-8]}_model_boxes'] = predictions
            predictions_list.append(predictions)

        # Get ground truth boxes
        row['true_boxes'] = load_ground_truth(label_path)
        # Ensemble predictions
        row['average_ensemble'] = average_ensemble(predictions_list)
        # row['max_confidence_ensemble'] = max_confidence_ensemble(predictions_list)
        # row['nms_ensemble'] = nms_ensemble(predictions_list)

        # Append to DataFrame
        df.loc[image_id-1]= row
        image_id += 1

# Save DataFrame to CSV
df.to_csv('predictions_labels_with_average_ensemble.csv', index=False)
