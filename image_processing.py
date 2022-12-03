from glob import glob
import os
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# All files and directories ending with .txt and that don't begin with a dot:

letter = "E"
IMAGE_FILES = glob("data/"+ letter + "/*.jpg")
path = 'data-mp/' + letter
#Confidence used for each letter A=.5,B=.5,C=.95,D=.95
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=.975) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_handedness == None:
      print ("removing:", file)
      os.remove(file)
        