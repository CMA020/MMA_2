from ultralytics import YOLO
import os
import cv2
import numpy as np

import argparse
import time
from collections import deque
from operator import itemgetter

from threading import Thread
out = None
msg="he"
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
import multiprocessing

from mmaction.apis import init_recognizer
stack1 = deque(maxlen=225)
model = YOLO(os.path.expanduser('/content/MMA_2/P_1920_30_3.pt'))
cap = cv2.VideoCapture(os.path.expanduser('/content/MMA_2/manc.mp4'))
#cap = cv2.VideoCapture(0)


img = cv2.imread(os.path.expanduser('/content/MMA_2/1.png'))

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file/url')
    # # parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=4,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results(queue):
    msg = "he"

    print('Press "Esc", "q" or "Q" to exit')
    #cap = cv2.VideoCapture(os.path.expanduser('~/catkin_ws/src/urdf_config7/scripts/manc.mp4'))
    text_info = {}
    cur_time = time.time()
    while True:
        ret, frame = cap.read()
        queue.put(frame)
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if selected_label == "goal":
                    if score < threshold:
                        msg = "no goal"
                        break
                    location = (0, 40 + i * 20)
                    text = selected_label + ': ' + str(round(score * 100, 2))
                    text_info[location] = text
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)
                    msg = text



        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)
        print(msg)
        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        scores = result.pred_scores.item.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

            if inference_fps > 0:
                # add a limiter for actual inference fps <= inference_fps
                sleep_time = 1 / inference_fps - (time.time() - cur_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cur_time = time.time()

def image_callback(queue):
    #cap = cv2.VideoCapture(os.path.expanduser('~/catkin_ws/src/urdf_config7/scripts/manc.mp4'))
    while True:

        #ret, frame = cap.read()
        img = queue.get()
        print("why")
        stack1.append(img)
def save(queue):


    while True:
        if (msg == "no"):

            print("printing no")
        if (msg == "goal"):
            time.sleep(2)
            print("buffering")
            flag = 1
            for img in stack1:
                out.write(img)
            out.release()
            cv2.waitKey(10)


# def predict(img):
#     cv2.imshow("img",img)
#     print("helo")
#     model = YOLO(os.path.expanduser('~/catkin_ws/src/urdf_config7/scripts/P_1920_30_3.pt'))
#     model.predict(img, save=True,show=True)
#
#     results = model(img)
#     for result in results:
#         boxes = result.boxes  # Boxes object for bbox outputs
#         masks = result.masks  # Masks object for segmentation masks outputs
#         keypoints = result.keypoints  # Keypoints object for pose outputs
#         probs = result.probs  # Probs object for classification outputs
#
#         for box in boxes:
#             # print(box.xywh)
#             if (int(box.cls) == 0):
#                 x = int(box.xywh[0][0])
#                 y = int(box.xywh[0][1])
#                 w = int(box.xywh[0][2])
#                 h = int(box.xywh[0][3])
#                 i = int(box.cls)
#
#                 print("helo")
#                 # msg = c
#                 # pub.publish(msg)
#
#                 # rate.sleep()


def camera1(queue):
    frame_counter=0
    cap = cv2.VideoCapture(os.path.expanduser('/content/MMA_2/manc.mp4'))
    #cap = cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        #frame = queue.get()
        cv2.imshow("br",frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        print("camera1")
        frame_counter += 1  # Increment the frame counter

        # Process the frame only when the counter is divisible by 3
        if frame_counter % 3 == 0:

            model = YOLO(os.path.expanduser('/content/MMA_2/P_1920_30_3.pt'))

            # model.predict(frame, save=True, show=True)
            # cv2.waitKey(1)
            results=model(frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    queue = multiprocessing.Queue()
    #cap = cv2.VideoCapture(os.path.expanduser('~/catkin_ws/src/urdf_config7/scripts/manc.mp4'))
    #cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    i = 0
    # Get the camera frame rate
    fps = 15
    out = None
    stack1 = deque(maxlen=225)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use XVID codec
    output_directory = os.path.expanduser("~/buffer")
    os.makedirs(output_directory, exist_ok=True)

    # Generate a unique file name for the output video
    output_filename = "buffer_video.mkv"
    output_path = os.path.join(output_directory, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    global average_size, threshold, drawing_fps, inference_fps, \
        device, model, camera, data, label, sample_length, \
        test_pipeline, frame_queue, result_queue

    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    device = args.device

    cfg = Config.fromfile(os.path.expanduser('/content/MMA_2/demo_config.py')
                          )
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, os.path.expanduser('/content/MMA_2/epoch_12l.pth')
                            , device=args.device)
    #camera = cv2.VideoCapture(args.camera_id)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(os.path.expanduser('/content/MMA_2/label_names.txt'), 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:

        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = multiprocessing.Process(target=show_results)

        pw = Thread(target=show_results, args=(queue,), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        queue = multiprocessing.Queue()

        r = multiprocessing.Process(target=camera1, args=(queue,))
        s = multiprocessing.Process(target=image_callback, args=(queue,))
        m = multiprocessing.Process(target=save, args=(queue,))

        r.start()
        s.start()
        m.start()

        # pw.start()
        # pr.start()




        #pw.join()
    except KeyboardInterrupt:
        pass







if __name__ == '__main__':
    main()


