import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import math

model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def detect_keypoints(image):
    input_image = tf.image.resize(image, [192, 192])
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    outputs = model.signatures['serving_default'](input_image)
    keypoints = outputs['output_0']
    return keypoints.numpy()[0]

def draw_keypoints_and_skeleton(image, keypoints, confidence_threshold=0.3):
    height, width, _ = image.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (0, 9), 
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (0, 15), (15, 16)
    ]
    for keypoint in keypoints:
        y, x, confidence = keypoint
        if confidence > confidence_threshold:
            cv2.circle(image, (int(x * width), int(y * height)), 4, (0, 255, 0), -1)

    for (start, end) in connections:
        start_point = keypoints[start]
        end_point = keypoints[end]
        if start_point[2] > confidence_threshold and end_point[2] > confidence_threshold:
            start_x, start_y = int(start_point[1] * width), int(start_point[0] * height)
            end_x, end_y = int(end_point[1] * width), int(end_point[0] * height)
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    return image

def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang

def draw_text(image, angles):
    height, width, _ = image.shape
    y_offset = height - 20  
    for name, angle in angles:
        text = f'{name}: {int(angle)}'
        cv2.putText(
            image, text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        y_offset -= 30  

video_path = 'swim2.mov'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(video_path+'_output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    keypoints = detect_keypoints(rgb_frame)
    keypoints = keypoints[0]
    
    frame_with_keypoints = draw_keypoints_and_skeleton(frame, keypoints)
    
    relevant_pairs = [
        ((5, 6, 7), "Left Elbow"), 
        ((6, 5, 11), "Left Shoulder"),
        ((11, 5, 12), "Right Shoulder"),
        ((12, 11, 13), "Right Elbow"),
        ((6, 5, 11), "Left Knee"),
        ((11, 5, 12), "Right Knee")
    ]

    angles = []
    for (a, b, c), name in relevant_pairs:
        if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3 and keypoints[c][2] > 0.3:
            angle = calculate_angle(keypoints[a], keypoints[b], keypoints[c])
            angles.append((name, angle))

    draw_text(frame_with_keypoints, angles)
    
    out.write(frame_with_keypoints)

cap.release()
out.release()

