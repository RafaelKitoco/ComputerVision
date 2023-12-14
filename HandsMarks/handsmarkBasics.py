import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
handsMarks = mp.solutions.hands
hands = handsMarks.Hands()
handsdraw = mp.solutions.drawing_utils

ctime = 0
ptime = 0

while True:
    sucess, img = cap.read()
    imageCV2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = hands.process(imageCV2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, ln in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int((ln.x*w)), int((ln.y*h))
                cv2.circle(img,(cx, cy), 5, (255,0,255), cv2.FILLED)
            handsdraw.draw_landmarks(img, hand, handsMarks.HAND_CONNECTIONS)
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 3)
    cv2.imshow("img", img)
    cv2.waitKey(1)