import mediapipe as mp
import cv2
import time

class HandmarksDetection():
    def __init__(self):
        self.handsMarks = mp.solutions.hands
        self.hands = self.handsMarks.Hands()
        self.handsdraw = mp.solutions.drawing_utils
    
    def findhands(self, img, draw = True):
        imageCV2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.result = self.hands.process(imageCV2)

        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    self.handsdraw.draw_landmarks(img, hand, self.handsMarks.HAND_CONNECTIONS)
        return img
    
    def findpositions(self, img, handNum=0, draw=True):
        listHnad = []
        if self.result.multi_hand_landmarks:
            myhands = self.result.multi_hand_landmarks[handNum]
            for id, ln in enumerate(myhands.landmark):
                h, w, c = img.shape
                cx, cy = int((ln.x*w)), int((ln.y*h))
                listHnad.append([id, ln])
                if draw:
                    cv2.circle(img,(cx, cy), 5, (255,0,255), cv2.FILLED)
        return listHnad

def main():
    cap = cv2.VideoCapture(0)
    ctime = 0
    ptime = 0
    detected = HandmarksDetection()
    while True:
        sucess, img = cap.read()
        img = detected.findhands(img)
        position = detected.findpositions(img)
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 3)
        cv2.imshow("img", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()