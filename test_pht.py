import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

image = cv2.imread("istockphoto-462795637-612x612.jpg")
image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
hands = mp_hands.Hands()
results = hands.process(image)
if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
print(results.multi_hand_landmarks)
print()
print(mp_hands.HAND_CONNECTIONS)
cv2.imshow('Handtracker', image)
cv2.waitKey()

