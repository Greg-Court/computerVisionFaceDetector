import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm) - classifer means detector, and cascade is the algorithm it uses
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('elon.jpeg')
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
  # read the current frame
  successful_frame_read, frame = webcam.read()

  # convert image to greyscale
  greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # get coordinates of face
  face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

  # Draw rectange around the faces
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
  
  #show the image and keep it open till any key is pressed
  cv2.imshow('Face Detector App', frame)
  key = cv2.waitKey(1)

  if key == 81 or key == 113:
    break

webcam.release()