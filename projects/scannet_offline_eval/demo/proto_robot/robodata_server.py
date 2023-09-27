import logging
from concurrent import futures

import grpc
import robodata_pb2
import robodata_pb2_grpc


class RoboDataServicer(robodata_pb2_grpc.RobotDataServicer):
    def GetHistory(self, request, context):
        if request is not None:
            x = robodata_pb2.RobotSummary()
            x.message = "hello"
            return x

    def ReceiveRobotData(self, request, context):
        print("entered ReceiveRobotData")
        print(request)
        x = robodata_pb2.RobotSummary()
        x.message = "hello response"
        # robo_obs = robodata_pb2.RoboObs()
        # robo_obs.gps = robodata_pb2.GPS_arr()
        x.robot_obs.gps.lat = 133.1415
        x.robot_obs.gps.long = 256.1415
        print(x)
        yield x


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    robodata_pb2_grpc.add_RobotDataServicer_to_server(RoboDataServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("started RoboData server")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
