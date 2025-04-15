# fingers
import cv2 as cv
import numpy as np
import mediapipe as mp
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
# mp_fase_mesh = mp.solusion.face_mesh
mp_hands = mp.solutions.hands

# Output Images
os.makedirs('Output Images', exist_ok=True)

video = cv.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:

    while video.isOpened():
        ret, frame = video.read()  # ret for return?
        #detection
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = cv.flip(img, 1)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        print(results)
        mask = np.zeros_like(frame)  # creating mask
        if results.multi_hand_landmarks:  # check, if hand caought
            # for num, hand, in enumerate(results.multi_hand_landmarks):  # enumerate acces to num and hand, multi_hand_landmarks is nums and hand  
            for hand in results.multi_hand_landmarks:    
                mp_drawing.draw_landmarks(mask, hand, mp_hands.HAND_CONNECTIONS, # mask to me and backgraund, img to picture the video
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness = 2, circle_radius=4), # joints
                                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness = 4, circle_radius=2), # connection betwin joints
                                        )
                cv.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), img)
        results_img = cv.bitwise_and(frame, mask)
        
        cv.imshow('hand traking', results_img)
        cv.imshow('only image', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
video.release()
cv.destroyAllWindows()

