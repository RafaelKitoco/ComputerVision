import cv2
import mediapipe as mp

cap = cv2.VideoCapture("C:\\Users\\pc\\Desktop\\ComputerVision\\ComputerVision\\video\\v.mp4") #colocar aqui o caminho para o seu video
postmark = mp.solutions.pose
pose = postmark.Pose()
drawpose = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Erro ao abrir o vídeo")
else:
    while True:
        success, img = cap.read()

        if not success:
            break
        imageCV2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(imageCV2)
        if result.pose_landmarks:
            for id, ln in enumerate(result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(ln.x*w), int(ln.y*h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), 3)
            drawpose.draw_landmarks(img, result.pose_landmarks, postmark.POSE_CONNECTIONS)

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
