import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import gazebo_msgs.msg as gazeboMsgs

class GazeboCamera():

    # cameraTopicPath = '/inventacao_car/camera1/image_raw'
    def __init__(self, cameraTopicPath):
        super().__init__()
        self.opencvBridge = CvBridge()
        self.onReceiveImage = None
        self.cameraTopicPath = cameraTopicPath
        self.frontCAM_sub = rospy.Subscriber(self.cameraTopicPath, Image, self.processImage)

    def setOnReceiveImage(self, onReceiveImageFunction):
        self.onReceiveImage = onReceiveImageFunction

    def processImage(self, ros_image):
        # Exit function in case ros stopped
        if (rospy.is_shutdown()):
            return
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.opencvBridge.imgmsg_to_cv2(ros_image, "bgr8")
            if (self.onReceiveImage != None):
                self.onReceiveImage(frame)
        except CvBridgeError as e:
            print ('CvBridgeError: ' + str(e))
            return

class RosInterface():

    def __init__(self, nodeName):
        super().__init__()
        rospy.init_node(nodeName)

    def listenToCamera(self, cameraTopicPath, onReceiveImageFunction):
        self.gazeboCamera = GazeboCamera(cameraTopicPath)
        self.gazeboCamera.setOnReceiveImage(onReceiveImageFunction)

    def listenToModelsStates(self,onReceiveModelsStateFunction):
        self.subscriberModelStates = rospy.Subscriber("/gazebo/model_states",
            gazeboMsgs.ModelStates, onReceiveModelsStateFunction)

    @staticmethod
    def getModelState(modelName, modelStates):
        for i in range(len(modelStates.name)):
            if modelStates.name[i] == modelName:
                state_dict = {
                    'name': modelStates.name[i],
                    'pose': modelStates.pose[i],
                    'twist': modelStates.twist[i]

                }
                return state_dict

