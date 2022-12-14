import rospy
import imagiz
import cv2

server = imagiz.TCP_Server(port=9990)
server.start()
while True:
    message = server.receive()
    frame = cv2.imdecode(message.image, 1)
    cv2.imshow("", frame)
    cv2.waitKey(1)
