import cv2
import time
import numpy as np
import argparse
import torch
import utils
from torchvision import transforms, models

model = models.mobilenet_v2()
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(1280, 29))
model.load_state_dict(torch.load('gesturenet_weights', map_location=torch.device('cpu')))
model.eval()
# model = torch.compile(model)

idx2gesture = ['fist', 'palm', 'other', 'ok', 'other', 'fist', 'ok', 'other',
    'other', 'other', 'other', 'peace', 'finger', 'other', 'other', 'other',
    'ok', 'ok', 'other', 'other', 'fist', 'other', 'fist', 'finger', 'peace',
    'palm', 'other', 'peace', 'finger']
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
# detection_graph, sess = utils.load_inference_graph()

def classify(img): #expects 128*128*3 input
    model_input = torch.unsqueeze(transform(img), dim=0)
    return idx2gesture[model(model_input).argmax()]

def main(args):
    cap = cv2.VideoCapture(int(args.video_source))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    im_width, im_height = (cap.get(3), cap.get(4))

    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)

    GESTURE_MEMORY_LENGTH = 6
    gesture_memory = ['other']*GESTURE_MEMORY_LENGTH

    while True:
        _, image_bgr = cap.read()
        image_bgr = cv2.flip(image_bgr, 1)    # reduces fps heavily
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_canvas = image_bgr
        face_seen = True

        # boxes, scores = utils.detect_objects(image_np, detection_graph, sess)
        # box, score = boxes[0], scores[0]
        box, score = np.array([0, 0, 1, 1]), 0.5

        if score > 0: #args.score_thresh:
            box = utils.adjust_bounding_box(box, im_width, im_height)
            pred = utils.draw_box_and_classify(box, image_bgr, classify,
                im_width, im_height, overlay_canvas, False)
            gesture_memory.append(pred)
            gesture_memory = gesture_memory[1:]

        #decide what to draw
        def count_then_recency(ges):
            if ges == 'other':
                return 0
            c = 0
            for i, g in enumerate(gesture_memory):
                if g == ges:
                    c += 1 + i*0.01
            return c
        current_gesture = max(gesture_memory, key=count_then_recency)
        utils.draw_overlay_image(current_gesture, overlay_canvas)

        cv2.imshow('Overlay', overlay_canvas)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # throttle the FPS
        time.sleep(0.05)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '--src',
        dest='video_source',
        default=1,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=500,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=275,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=8,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()
    
    main(args)