import argparse

import cv2
import torch
import time

from mmdet.apis import inference_detector, init_detector, show_result, show_result_ins


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)
    count = 0
    time_start = time.monotonic()
    print(time.monotonic())

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        img_show = show_result_ins(img, result, model.CLASSES, score_thr=0.25)
        cv2.imshow('Demo', img_show)
        if (count<=10):
            count+=1
        else:
            time_end = time.monotonic()
            frame_time = (time_end-time_start)/10
            fps = 1/frame_time
            print("fps=",fps)
            time_start = time_end
            count = 0


        #print("--",time.monotonic())


if __name__ == '__main__':
    main()
