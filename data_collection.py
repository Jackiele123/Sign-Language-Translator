import cv2
import mediapipe as mp
import numpy as np
import math
import time
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

letter = "E"

offset = 25
imgSize = 200

folder = "data/" + letter
counter = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    results = hands.process(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    h, w, c = image.shape

    if results.multi_hand_landmarks:

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        for hand_landmarks in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cropped_image = image[y_min-25:y_max+25, x_min-25:x_max+25]
            # mp_drawing.draw_landmarks(
            # image,
            # hand_landmarks,
            # mp_hands.HAND_CONNECTIONS,
            # mp_drawing_styles.get_default_hand_landmarks_style(),
            # mp_drawing_styles.get_default_hand_connections_style())

        height, width, c = cropped_image.shape

        aspectRatio = height / width
        
        if aspectRatio > 1:
            k = imgSize / height
            wCal = math.floor(k * width)
            imgResize = cv2.resize(cropped_image, (wCal, imgSize))
            wGap = math.floor((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / width
            hCal = math.floor(k * height)
            imgResize = cv2.resize(cropped_image, (imgSize, hCal))
            hGap = math.floor((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_' + letter + '_' + str(counter) + '.jpg',imgWhite)
        print(counter)

    if cv2.waitKey(5) & 0xFF == ord('r'):
      break

cap.release()