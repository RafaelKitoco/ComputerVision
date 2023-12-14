import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
handsMarks = mp.solutions.hands
hands = handsMarks.Hands()
handsdraw = mp.solutions.drawing_utils

while True:
    sucess, img = cap.read()
    imageCV2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = hands.process(imageCV2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            handsdraw.draw_landmarks(img, hand)

    cv2.imshow("img", img)
    cv2.waitKey(1)