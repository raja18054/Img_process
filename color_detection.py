import cv2
import numpy as np
import argparse
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to input image')
parser.add_argument('--webcam', type=int, nargs='?', const=0, help='Use webcam. Optionally pass camera index (default 0)')
args = parser.parse_args()

COLOR_RANGES = [
    ('Red',    (0, 120, 70),   (10, 255, 255), (0,0,255)),
    ('Red2',   (170,120,70),   (180,255,255), (0,0,255)),
    ('Green',  (36,  50, 70),   (89, 255, 255), (0,255,0)),
    ('Blue',   (90,  50, 70),   (128, 255,255), (255,0,0)),
    ('Yellow', (15, 150, 150),  (35, 255,255), (0,255,255)),
]

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

def detect_colors(frame):
    annotated = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for name, lower, upper, disp_color in COLOR_RANGES:
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w//2
            cy = y + h//2
            cv2.rectangle(annotated, (x,y), (x+w, y+h), disp_color, 2)
            label = 'Red' if name == 'Red2' else name
            cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, disp_color, 2)
            cv2.circle(annotated, (cx, cy), 4, disp_color, -1)
    return annotated

if args.image:
    img = cv2.imread(args.image)
    if img is None:
        print('Error: could not load image:', args.image)
        exit(1)
    resized = imutils.resize(img, width=800)
    out = detect_colors(resized)
    cv2.imshow('Color Detection - Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args.webcam is not None:
    cam_index = 0 if args.webcam is None else args.webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Cannot open webcam', cam_index)
        exit(1)
    print('Press q to quit, s to save a screenshot')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=800)
        out = detect_colors(frame)
        cv2.imshow('Color Detection - Webcam', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('color_detection_capture.jpg', out)
            print('Saved color_detection_capture.jpg')
    cap.release()
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('No webcam detected. Run with --image path/to/image.jpg')
        exit(1)
    print('Press q to quit, s to save a screenshot')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=800)
        out = detect_colors(frame)
        cv2.imshow('Color Detection - Webcam', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('color_detection_capture.jpg', out)
            print('Saved color_detection_capture.jpg')
    cap.release()
    cv2.destroyAllWindows()
