import cv2
import numpy as np


def load_net():
    """Loads YOLOv3 configuration to OpenCV"""
    return cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


if __name__ == '__main__':
    classes = [line.rstrip() for line in open('yolo/coco.names').readlines()]
    count_of_calls = dict.fromkeys(classes, 0)
    confidence_of_calls = dict.fromkeys(classes, 0)
    net = load_net()
    video = cv2.VideoCapture('rgb_video22.mp4')
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    new_video = cv2.VideoWriter('final.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                15, (640, 480), True)

    _, img = video.read()
    while img is not None:
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'people':
                    continue
                confidence = str(round(confidences[i], 2))
                count_of_calls[label] += 1
                confidence_of_calls[label] += round(confidences[i], 2)
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        new_video.write(img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        _, img = video.read()

    for names_label in classes:
        if count_of_calls[names_label] != 0:
            print(names_label, ": count = ", count_of_calls[names_label], \
                  " confidence = ", toFixed(confidence_of_calls[names_label] / count_of_calls[names_label], 3))

    video.release()
    new_video.release()
    # cv2.destroyAllWindows()
