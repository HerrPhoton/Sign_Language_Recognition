import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
all_list = []
iter_f = 0
for filename in os.listdir('document_5434070765534195166.mp4-opencv'):
    fil_n = "document_5434070765534195166.mp4-opencv/" + filename
    print(fil_n)
    image = cv2.imread(fil_n)
    # cv2.imshow('Mediapipe Feed', image)

    ## Setup mediapipe instance
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # while image.isOpened():
        #     ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = holistic.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # cv2.imshow('Mediapipe Feed', image)
        # cv2.waitKey()

        # cap.release()
        # cv2.destroyAllWindows()
    # print(mp_holistic.POSE_CONNECTIONS)
    leng = len(results.pose_landmarks.landmark)

    for itr in range(leng):
        lst = [iter_f, 'pose', itr, results.pose_landmarks.landmark[itr].x,
               results.pose_landmarks.landmark[itr].y, results.pose_landmarks.landmark[itr].z]
        all_list.append(lst)
    iter_f += 1
df = pd.DataFrame(all_list, columns=['frame', 'type', 'landmark_index', 'x', 'y', 'z'])

print(df)
