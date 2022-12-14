import rospy
import imagiz

server = imagiz.Server()
while True:
    message = server.recive()
    frame = cv2.imdecode(message.image, 1)
    cv2.imshow("", frame)
    cv2.waitKey(1)
