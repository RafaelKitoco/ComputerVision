import cv2
import mediapipe as mp

class Posemudule():
        def __init__(self) -> None:
            self.postmark = mp.solutions.pose
            self.pose = self.postmark.Pose()
            self.drawpose = mp.solutions.drawing_utils
        
        def findpose(self, img, draw = True):
            imageCV2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.result = self.pose.process(imageCV2)
            
            if self.result.pose_landmarks:
                if draw:
                    self.drawpose.draw_landmarks(img, self.result.pose_landmarks, self.postmark.POSE_CONNECTIONS)
            return img
        
        def findpositions(self, img, PoseNum=0, draw = True):
            listpose = []
            
            if self.result.pose_landmarks:
                mypose = self.result.pose_landmarks
                for id, ln in enumerate(mypose.landmark):
                        h, w, c = img.shape
                        cx, cy = int(ln.x*w), int(ln.y*h)
                        listpose.append([id, ln])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), 3)
            return listpose
    
def main():
    cap = cv2.VideoCapture("C:\\Users\\pc\\Desktop\\ComputerVision\\ComputerVision\\videos\\v.mp4") #colocar aqui o caminho para o seu video
    detected = Posemudule()
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
    else:
        while True:
            success, img = cap.read()
            img = detected.findpose(img)
            position = detected.findpositions(img)

            if not success:
                break

            cv2.imshow("image", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()