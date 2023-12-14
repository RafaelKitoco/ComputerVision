import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
handsMarks = mp.solutions.hands
hands = handsMarks.Hands()


while True:
    sucess, img = cap.read()

    cv2.imshow("img", img)
    cv2.waitKey(1)