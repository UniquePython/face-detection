import sketch
import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source/camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

window_name = 'Live Sketch'
cv2.namedWindow(window_name)

def nothing(x): pass

cv2.createTrackbar('Blur Strength', window_name, 1, 255, nothing)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.getTrackbarPos('Blur Strength', window_name)
    cv2.imshow(window_name, sketch.generate_sketch_from_array(gray_frame, blur / 255))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()