import cv2
import torch
import logging
import requests
import argparse
import numpy as np
from datetime import datetime

import aiohttp
import asyncio
import socket

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


requests.packages.urllib3.util.connection.HAS_IPV6 = False

logging.basicConfig(filename="feeding_events.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s - %(message)s',
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
file_logger = logging.getLogger(name="Feeding events")



def send_notification(message):
    url = "https://api.pushover.net/1/messages.json"


    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    message = now + " - " + message

    payload = {
        "token": "ausacuqa5urtmhytuc8of2tak72cwa",
        "user": "u957ugfwotcdsngjumoqyruukhfkr2",
        "message": message 
    }

    file_logger.info("Sending notification")
    r = requests.post(url=url, data=payload)
    file_logger.info(f"Notification sent with response: {r}")


def write_to_file(logger, message: str):
    logger.info(message)


def check_intersection(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_intersect_min = max(x1_min, x2_min)
    y_intersect_min = max(y1_min, y2_min)
    x_intersect_max = min(x1_max, x2_max)
    y_intersect_max = min(y1_max, y2_max)

    width_intersect = x_intersect_max - x_intersect_min
    height_intersect = y_intersect_max - y_intersect_min

    if width_intersect > 0 and height_intersect > 0:
        return True
    else:
        return False


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--ip', type=str)
    parser.add_argument('--stream_nr', type=str)
    opt = parser.parse_args()
    return opt


@smart_inference_mode()
def process_video(username, password, ip, stream_nr, weights='yolov9-m.pt', image_size=(360, 640)):
    last_person_write = None
    last_dog_write = None
    
    # Other params
    augment = False
    conf_thres = 0.35    # confidence threshold
    iou_thres = 0.45     # NMS IOU threshold
    max_det = 1000       # maximum detections per image
    agnostic_nms = False # class-agnostic NMS
    classes = [0, 16]       # filter by class: --class 0, or --class 0 2 3
    view_img = True      # show results
    save_txt = False     # save results to *.txt
    save_conf = False    # save confidences in --save-txt labels
    save_crop = False    # save cropped prediction boxes
    line_thickness = 2   # bounding box thickness (pixels)
    update = False        # update all models

    feeding_area = {
        "label": "feeding area",
        "coords": ((270, 749), (362, 843)),
        "color": (0, 255, 0),
    }

    # Alert Params
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 0, 255)  # Red
    font_thickness = 3
    # position = (10, 30)  # Top-left corner

    device = select_device("0")
    model = DetectMultiBackend(weights, device, dnn=False, fp16=True)
    stride, names, pt = model.stride, model.names, model.pt

    print(f"Using device: {device}")

    imgsz = check_img_size(image_size, s=stride)

    source = f"rtsp://{username}:{password}@{ip}/stream{stream_nr}"

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)

    fps = int(dataset.fps[0])
    person_intersection_frame_count = [0] * fps * 6
    dog_intersection_frame_count = [0] * fps * 6

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    #  Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    frame_idx = 0
    for path, im, im0s, vid_cap, s in dataset:
        if frame_idx < (fps * 6) - 1:
            frame_idx += 1
        else:
            frame_idx = 0

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # Check intersection with feeding area
                    feeding_area_xyxy = [item for subtuple in feeding_area["coords"] for item in subtuple]
                    if check_intersection(xyxy, feeding_area_xyxy):
                        if c == 0:
                            person_intersection_frame_count[frame_idx] = 1
                            if sum(person_intersection_frame_count) >= 60:
                                message = "Person intersection"
                                # Calculate the text size to determine the position
                                text_size, _ = cv2.getTextSize(message, font, font_scale, font_thickness)
                                text_width, text_height = text_size
                                position = (im0.shape[1] - text_width - 10, text_height + 10) # Top-right corner
                                im0 = cv2.putText(im0, message, position, font, font_scale, font_color, font_thickness)
                                
                                if last_person_write is None or (datetime.now() - last_person_write).total_seconds() > 60:
                                    write_to_file(file_logger, "Person in feeding area")
                                    last_person_write = datetime.now()
                                    send_notification("Person in feeding area")

                        elif c == 16:
                            dog_intersection_frame_count[frame_idx] = 1
                            if sum(dog_intersection_frame_count) >= 60:
                                message = "Dog intersection"
                                # Calculate the text size to determine the position
                                text_size, _ = cv2.getTextSize(message, font, font_scale, font_thickness)
                                text_width, text_height = text_size
                                position = (im0.shape[1] - text_width - 10, text_height + 10) # Top-right corner
                                im0 = cv2.putText(im0, message, position, font, font_scale, font_color, font_thickness)

                                if last_dog_write is None or (datetime.now() - last_dog_write).total_seconds() > 60:
                                    write_to_file(file_logger, "Dog in feeding area")
                                    last_dog_write = datetime.now()
                                    send_notification("Dog in feeding area")
                    else:
                        person_intersection_frame_count[frame_idx] = 0
                        dog_intersection_frame_count[frame_idx] = 0
                    
            # Stream results
            im0 = annotator.result()
            annotator.box_label(
                [item for subtuple in feeding_area["coords"] for item in subtuple], 
                feeding_area["label"], 
                color=feeding_area["color"])
            
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])
if __name__ == "__main__":
    write_to_file(file_logger, "Started monitoring...")
    opt = parse_opt()
    process_video(**vars(opt))
    