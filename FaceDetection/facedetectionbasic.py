import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("C:\\Users\\pc\\Desktop\\ComputerVision\\ComputerVision\\videos\\face1.mp4")
mpFace = mp.solutions.face_detection
Facedetection = mpFace.FaceDetection() 
ptime = 0

if not cap.isOpened():
    print("Erro ao abrir o vídeo")
else:
    while True:
        sucess, img = cap.read()
        imageCV2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)               
        if not sucess:
             break
        result = Facedetection.process(imageCV2)
        if result.detections:
             for id, detection in enumerate(result.detections):
                  #print(id, detection)  
                  bbdc = detection.location_data.relative_bounding_box
                  ih, iw, ic = img.shape
                  bbb = int(bbdc.xmin*iw), int(bbdc.ymin*ih), int(bbdc.width*iw), int(bbdc.height*ih)
                  cv2.rectangle(img, bbb, (255, 0, 255), 2)

        ctime = time.time()
        fps = 1/(ctime- ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)                  
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

