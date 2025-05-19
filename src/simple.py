# from djitellopy import Tello
# import cv2

# tello = Tello()
# tello.connect()
# tello.streamon()

# frame_read = tello.get_frame_read()

# while True:
#     frame = frame_read.frame
#     cv2.imshow("Tello Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# tello.streamoff()
# cv2.destroyAllWindows()

import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()
cv2.imwrite("picture.png", frame_read.frame)

tello.land()