import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
# Dont forget to source the ROS enronment before running this script
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry

from gazebo_msgs.srv import SetModelState, SetModelStateRequest

from scipy.spatial.transform import Rotation

from time import time

from classes.ros_gazebo import GazeboCamera, RosInterface

class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        """
        Allow initialization with explicit quaterion wxyz, axis-angle, or Euler XYZ (RPY) angles.

        :param w: w (real) of quaternion.
        :param x: x (i) of quaternion.
        :param y: y (j) of quaternion.
        :param z: z (k) of quaternion.
        :param axis_angle: Set of three values from axis-angle representation, as list or [3,] or [3,1] np.ndarray.
                           See C2M5L2 for details.
        :param euler: Set of three XYZ Euler angles. 
        """
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle or euler can be specified.")
        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be list or np.ndarray with length 3.")
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-50:  # to avoid instabilities and nans
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()
        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            # Fixed frame
            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

            # Rotating frame
            # self.w = cr * cp * cy - sr * sp * sy
            # self.x = cr * sp * sy + sr * cp * cy
            # self.y = cr * sp * cy - sr * cp * sy
            # self.z = cr * cp * sy + sr * sp * cy

    def __repr__(self):
        return "Quaternion (wxyz): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.w, self.x, self.y, self.z)

    def to_axis_angle(self):
        t = 2*np.arccos(self.w)
        return np.array(t*np.array([self.x, self.y, self.z])/np.sin(t/2))

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - np.dot(v.T, v)) * np.eye(3) + \
               2 * np.dot(v, v.T) + 2 * self.w * self.skew_symmetric(v)

    def to_euler(self):
        """Return as xyz (roll pitch yaw) Euler angles."""
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """Return numpy wxyz representation."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        """Return a (unit) normalized version of this quaternion."""
        norm = np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def quat_mult_right(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the right, that is, q*self.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = -self.skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def quat_mult_left(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the left, that is, self*q.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = self.skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def skew_symmetric(self, v):
        """Skew symmetric form of a 3x1 vector."""
        return np.array(
            [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]], dtype=np.float64)

class OpencvTrackBars():
    def __init__(self, hueMin, hueMax,):
        super().__init__()

    def initCV(self):
        global rmin, rmax, bmin, bmax, gmin, gmax
        cv2.namedWindow( self.trackbarName )
        cv2.namedWindow(self.windowName, 1)
        cv2.namedWindow("Bottom", 1)
        
        # create trackbars for color change
        cv2.createTrackbar('Hmin',self.trackbarName,0,179, self.trackBar)
        cv2.createTrackbar('Hmax', self.trackbarName,0,179,self. trackBar)
        cv2.createTrackbar('Smin',self.trackbarName,0,255, self.trackBar)
        cv2.createTrackbar('Smax',self.trackbarName,0,255, self.trackBar)	
        cv2.createTrackbar('Vmin',self.trackbarName,0,255, self.trackBar)
        cv2.createTrackbar('Vmax',self.trackbarName,0,255, self.trackBar)
        
        cv2.createTrackbar('Threshmin',self.trackbarName,0,255, self.trackBar)
        cv2.createTrackbar('Threshmax',self.trackbarName,0,255, self.trackBar)
        
        cv2.setTrackbarPos( 'Threshmax', self.trackbarName, 50)
        cv2.setTrackbarPos( 'Hmax', self.trackbarName, 75)
        cv2.waitKey(1)

class Marker():
    def __init__(self, centerX, centerY):
        super().__init__()
        self.centerX = centerX
        self.centerY = centerY

    def toTuple(self):
        return (self.centerX, self.centerY)

    def __str__(self):
        return str(self.toTuple())



class InventacaoCar():
    
    def __init__(self, model_name, persistentSetModelState = False):
        super().__init__()
        self.orientation = {'x': 0, 'y': 0, 'z':0, 'w': 0}
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.modelName = model_name

        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistentSetModelState)
        self.objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest

        self.leftMarker = Marker(52+90, 523)
        self.rightMarker = Marker(745-90, 523)

        self.frame_width = -1
        self.frame_height = -1

        self.last_time = time()
        self.OriError = 0
        

    def onReceiveNavData(self):
        pass

    def onReceiveImuData(self, models_state):
        imuData = RosInterface.getModelState("inventacao_car_camera_light", models_state)
        print(imuData)
        if imuData == None:
            return
        pose = imuData['pose']
        twist = imuData['twist']
        #print(imuData)
        #print(pose)
        #print(type(pose))
        #print( 'position: ', pose.position)
        #print( 'orientation: ', pose.orientation)
        self.orientation = pose.orientation
        self.position = pose.position
        self.linear_velocities = twist.linear
        self.angular_velocities = twist.angular

    def onImageReceived(self, frame):

        if self.frame_width <= 0:
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]

        maxHeight = 450
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray,(5,5))
        thresh = np.ones_like(blur) * 255
        thresh = cv2.rectangle(thresh,(0,0),(thresh.shape[0],maxHeight),(0,0,0),-1)
        thresh[blur < 70] = 0
        thresh[blur > 80] = 0

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laneMarkers = []
        self.drawCornersMarkers(frame)
        middle = (self.frame_height/2,self.frame_width/2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print("Area: ", + area)
            if area < 2000:
                continue
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01'] / M['m00'])

            if cy < maxHeight:
                continue

            distanceToCenter = self.computeDistance((cx,cy),middle)
            #print('cx: ', cx, ' ****  cy: ', cy)
            #print('dist: ', distanceToCenter)

            #cv2.circle(frame,(cx,cy),9,(255,0,0),-1)
            
            laneMarkers.append({
                #'cnt': cnt,
                'centerX': cx,
                'centerY': cy,
                'area': area,
                'distanceToCenter': distanceToCenter
            })

        #for x in range(len(contours)):
        #cv2.drawContours(frame,contours,1,(0,255,0))

        frame_shape = frame.shape
        roadMarker = self.getRoadCenter(laneMarkers)
        #self.identifyLanesGivenContours(laneMarkers)
        cv2.circle(frame,roadMarker.toTuple(),10,(0,255,0),-1)

        #self.drawLaneMarkersPositions(frame)

        
        #cv2.circle(frame,(int(frame_shape[1]/2),int(frame_shape[0]/2)),10,(255,0,0),-1)
        #exit("fim")
        cv2.imshow("RobotFrontCameraOutput", thresh)
        
        cv2.imshow("RobotFrontCamera", frame)
        #distances = self.computeDistanceToCorners()
        #self.error = self.computeOrientationError(distances)
        self.error = self.frame_width/2 - roadMarker.centerX
        self.applyKinematics()
        #print('distances: ', distances)
        cv2.waitKey(30)
        print('\n\n\n')

    def computeOrientationError(self, distances):
        
        if distances[0] != None:
            return distances[0]
        else:
            return distances[1]

    def drawLaneMarkersPositions(self, frame):
        if self.leftLanePosition != None:
            cv2.circle(frame, (self.leftLanePosition.centerX,
            self.leftLanePosition.centerY), 9, (255, 0, 0), -1)
        if self.rightLanePosition != None:
            cv2.circle(frame, (self.rightLanePosition.centerX,
            self.rightLanePosition.centerY), 9, (0, 0, 255), -1)

    def computeDistanceToCorners(self):
        leftLaneDistance = None
        rightLaneDistance = None
        
        if( self.leftLanePosition != None ):
            leftLaneDistance = self.computeDistance(
                self.leftLanePosition.toTuple(), self.leftMarker.toTuple())
            
            if self.leftMarker.centerX < self.leftLanePosition.centerX:
                leftLaneDistance *= - 1
        
        if( self.rightLanePosition != None ):
            rightLaneDistance = self.computeDistance(
                self.rightLanePosition.toTuple(), self.rightMarker.toTuple())
            print(self.rightMarker)
            if self.rightMarker.centerX < self.rightLanePosition.centerX:
                rightLaneDistance *= - 1

        return (leftLaneDistance, rightLaneDistance)
                  
    def computeDistance(self, c1, c2):
        return ((c2[0] - c1[0])**2)**.5

    def getRoadCenter(self, contoursDict):
        contoursDict.sort(key=lambda x: x['distanceToCenter'])
        cnt = contoursDict[0]
        return Marker(cnt['centerX'], cnt['centerY'])

    def identifyLanesGivenContours(self, contoursDict):
        contoursDict.sort(key=lambda x: x['area'])
        laneMarkers = []

        for cnt in contoursDict:
            laneMarkers.append(Marker(cnt['centerX'], cnt['centerY']))
            # Accept the two first larger contours
            if (len(laneMarkers) > 1):
                break

        # Sort by X axis to ease left and right corners identification
        laneMarkers.sort(key=lambda x: x.centerX)

        self.leftLanePosition = None
        self.rightLanePosition = None
        if len(laneMarkers) == 2:
            self.leftLanePosition = laneMarkers[0]
            self.rightLanePosition = laneMarkers[1]
        elif len(laneMarkers) == 1:
            xLocation = laneMarkers[0].centerX
            frameCenterX = self.frame_width/2
            if xLocation > frameCenterX:
                self.rightLanePosition = laneMarkers[0]
            else:
                self.leftLanePosition = laneMarkers[0]

    def drawCornersMarkers(self, frame):
        frame_shape = frame.shape
        cv2.circle(frame, self.leftMarker.toTuple(), 15, (255, 0, 0), 3)
        cv2.circle(frame,self.rightMarker.toTuple(),15,(0,0,255),3)   
    
    def moveRobot(self, linearVelocities, angularVelocities):
        twist = {}
        twist['linear'] = linearVelocities
        twist['angular'] = angularVelocities

        r = Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1]))
        q = r.as_quat()

        # set red cube pose
        objstate = self.objstate
        objstate.model_state.model_name = self.modelName
        #objstate.model_state.pose.position.x = 0
        #objstate.model_state.pose.position.y = 0
        objstate.model_state.pose.position.z = 0.361203
        #objstate.model_state.pose.orientation.w = q[3]
        #objstate.model_state.pose.orientation.x = q[0]
        #objstate.model_state.pose.orientation.y = q[1]
        #objstate.model_state.pose.orientation.z = q[2]
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0
        objstate.model_state.reference_frame = "world"
        
        result = self.set_state_service(objstate)

    def computeOrientation(self):
        orientation = self.orientation
        dt = time() - self.last_time
        kp = 0.0008
        ki = 0.0001
        uOrientation = self.error * kp + self.OriError*ki*dt

        self.OriError += self.error

        r = Rotation.from_rotvec(uOrientation * np.array([0, 0, 1]))
        q = r.as_quat()
        q1 = Quaternion(q[3],q[0],q[1],q[2])
        q2 = Quaternion(orientation.w, orientation.x,
        orientation.y, orientation.z)

        res = q1.quat_mult_left(q2, out='Quaternion')
        return res

    def applyKinematics(self):
        from math import cos, sin
        orientation = self.orientation
        xVelocity = 3       # 1 m/s
        
        dt = time() - self.last_time
        print('Error: ', self.error)

        newOrientation = self.computeOrientation()

        r = Rotation.from_quat([newOrientation.x, newOrientation.y,
            newOrientation.z, newOrientation.w])

        print( '\nxyz\n', r.as_euler('xyz',degrees=False))

        zAngle = r.as_euler('xyz',degrees=False)[2]
        newX = self.position.x + xVelocity*sin(zAngle) * dt
        newY = self.position.y - xVelocity*cos(zAngle) * dt

        print('***********',newOrientation)

        objstate = self.objstate
        objstate.model_state.model_name = self.modelName
        objstate.model_state.pose.position.x =  newX
        objstate.model_state.pose.position.y = newY
        objstate.model_state.pose.position.z = 0.361203
        objstate.model_state.pose.orientation.w = newOrientation.w
        objstate.model_state.pose.orientation.x = newOrientation.x
        objstate.model_state.pose.orientation.y = newOrientation.y
        objstate.model_state.pose.orientation.z = newOrientation.z
        objstate.model_state.reference_frame = "world"
        
        result = self.set_state_service(objstate)

        self.last_time = time()

    def stopRobot(self):
        objstate.model_state.pose.position.x = 0.5
        objstate.model_state.pose.position.y = 0.5
        objstate.model_state.pose.position.z = 0.5
        objstate.model_state.pose.orientation.w = 1
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = 0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        self.moveRobot()

class PID():
    def __init__(self, kP, kI, kD, maxControlValue):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.maxControlValue = maxControlValue

        self.error = 0
        self.integratedErrors = 0
        self.lastError = 0
        self.lastUpdatedTime = -1

    def setSetPoint(self, setPoint):
        self.setPoint = setPoint

    def _computeError(self, currentState):
        return self.setPoint - currentState

    def _truncControlValue(self, u):
        maxVel = self.maxControlValue
        if u > maxVel:
            u = maxVel
        elif u < -maxVel:
            u = -maxVel

        return u

    def applyPID(self, currentState):
        if self.lastUpdatedTime == -1:
            self.lastUpdatedTime = time()
            return 0
        dt = time() - self.lastUpdatedTime
        error = self._computeError(currentState)

        deltaError = error - self.lastError
        u = error * self.kP \
            + self.integratedErrors * self.kI * dt \
                + (deltaError/dt)*self.kD
        #print("2PID error: ", error)
        self.integratedErrors += error
        self.lastError = error

        self.lastUpdatedTime = time()

        return self._truncControlValue(u)
    

class InventacaoCarPID():
    """xCoefficients={
        kp:value,
        kd:value,
        ki:value,
        maxControlValue

    }
    """
    def __init__(self, xCoefficients, yCoefficients,
    xLaneCoefficients, yLaneCoefficients, wCoefficients):
        self.PIDs: Dict[str,PID] = {}

        self._load_PID('x', xCoefficients)
        self._load_PID('y', yCoefficients)
        self._load_PID('xLane', xLaneCoefficients)
        self._load_PID('yLane', yLaneCoefficients)
        self._load_PID('w',wCoefficients)
        

    def _load_PID(self, name, coefficients):
        kP = coefficients['kP']
        kI = coefficients['kI']
        kD = coefficients['kD']
        maxControlValue = coefficients['maxControlValue']

        pid = PID(kP, kI, kD,maxControlValue)
        self.PIDs[name] = pid

    def setXSetPoint(self, x):
        self.PIDs['x'].setSetPoint(x)

    def setYSetPoint(self, y):
        self.PIDs['y'].setSetPoint(y)

    def setXLaneSetPoint(self, xLane):
        self.PIDs['xLane'].setSetPoint(xLane)

    def setYLaneSetPoint(self, yLane):
        self.PIDs['yLane'].setSetPoint(yLane)

    def setOmegaSetPoint(self, w):
        self.PIDs['w'].setSetPoint(w)

    def _setControlBoundary(self, u):
        maxVel = 1000
        if u > maxVel:
            u = maxVel
        elif u < -maxVel:
            u = -maxVel

        return u

    def applyPIDs(self, variablesCurrentState):
        variables = self.PIDs.keys()
        controls = {}
        for var in variables:
            #print("Reading PID: ", var)
            currentSate = variablesCurrentState[var]
            u = self.PIDs[var].applyPID(currentSate)            
            
            controls[var] = u

        return controls
        

class RotatedBox():
    def __init__(self, minRectAreaOutput):
        self.rect = minRectAreaOutput
        box = cv2.boxPoints(self.rect)
        # First element the point more below
        self.rotatedRect = np.int0(box)
        self.sides = self.indentifyVertices()
        self.center1, self.center2 = self.computeCenters(self.sides)

    def indentifyVertices(self):
        centerX, centerY = self.rect[0]
        lowerPoint = self.rotatedRect[0]
        lowerPointIndice = self.getPointIndex(lowerPoint,self.rotatedRect)
        closestPointToLowerPoint = self.getClosestPoint( lowerPoint, self.rotatedRect[1:] )
        closestPointIndice = self.getPointIndex(closestPointToLowerPoint, self.rotatedRect)
        side1 = [x for i, x in enumerate(self.rotatedRect) if i == lowerPointIndice or i == closestPointIndice]
        side2 = [ x for i,x in enumerate(self.rotatedRect) if i != lowerPointIndice and i != closestPointIndice ]
        #print('points: ', self.rotatedRect)
        #print('p1: ', self.rotatedRect[0])
        #print('p1 indice: ', lowerPointIndice)
        #print('closest point indice: ', closestPointIndice)
        #print('closestPoint: ', self.rotatedRect[closestPointIndice])
        #print('vert1: ', side1)
        #print('vert2: ', side2)
        
        return (side1, side2)

    def computeCenters(self, sides):
        side1 = sides[0]
        side2 = sides[1]
        
        p11 = side1[0]
        p12 = side1[1]

        p21 = side2[0]
        p22 = side2[1]

        center1 = (int((p11[0] + p12[0]) / 2), int((p11[1] + p12[1]) / 2))
        center2 = (int((p21[0] + p22[0]) / 2), int((p21[1] + p22[1]) / 2))
        
        return (center1, center2)

    def getClosestPoint(self, p1, points):
        sortedPoints = sorted(points, key=lambda x: self.computeDistance(p1, x),reverse=False)
        #print('sortedPoints: ', sortedPoints)
        #print('sortedClosestPoint: ', sortedPoints[0])
        return sortedPoints[0]

    def getPointIndex(self, p1, points):
        position = None
        for pos, point in enumerate(points):
            #print(pos, ' - dist: ', self.computeDistance(p1, point))
            if self.computeDistance(p1, point) == 0:
                position = pos
                break
        return position

    def computeDistance(self, c1, c2):
        return ((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)**.5

    def getCenters(self):
        return (self.center1, self.center2)


class QrCodeOrientation():
    def __init__(self, currentOrientaion):
        self.targetOrientation = currentOrientaion
        self.updateRateInFrames = 5
        self.currentSkippedFrames = self.updateRateInFrames

    def getTargetOrientation(self):
        return self.targetOrientation

    def checkAndReadQrCode(self, frame):
        import pyzbar.pyzbar as pyzbar
        import json

        if self.currentSkippedFrames < self.updateRateInFrames:
            self.currentSkippedFrames += 1
            return self.targetOrientation
        
        self.currentSkippedFrames = 0

        #gamma = 0.7
        #lookUpTable = np.empty((1,256), np.uint8)
        #for i in range(256):
        #    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        #res = cv2.LUT(np.array(res,copy=True), lookUpTable)        
        
        #cv2.imshow("QrDetector",res)
        # Find barcodes and QR codes
        decodedObjects = pyzbar.decode(frame)
        if len(decodedObjects) > 0:
            # Print results
            for obj in decodedObjects:
                print('Type : ', obj.type)
                print('Data : ', obj.data, '\n')
                str_data = obj.data.decode('utf-8')
                str_data = str_data.replace('{', '{"')
                str_data = str_data.replace(':', '":')
                print('str_data: ',str_data)
                data = json.loads(str_data)
                self.targetOrientation = np.deg2rad(data['turn'])
                #cv2.waitKey(0)
        else:
            print('No qrcode found')

        return self.targetOrientation

        #cv2.waitKey(0)

        

        """
        data, bbox, rectifiedImage = self.qrDecoder.detectAndDecode(frame)
        if len(data) > 0:
            print("Decoded Data : {}".format(data))
        else:
            print('No qrcode found!!!')
        """

class InventacaoCarCameraBelow(InventacaoCar):
    
    def __init__(self, model_name):
        super().__init__(model_name, False)
        
        self.PIDs = InventacaoCarPID({"kP": 0.01, "kD": 0, "kI": 0.00001, "maxControlValue": 1},
        {"kP": 0.01, "kD": 0, "kI": 0.00001, "maxControlValue": 1},
        {"kP": 0.001, "kD": 0, "kI": 0.006, "maxControlValue":.4},
        {"kP": 0.001, "kD": 0, "kI": 0.006, "maxControlValue":.4},
        {"kP":.5,"kD":0.1,"kI":0.0001, "maxControlValue":0.24})

        self.lastTopCenter = None

        self.bottomLightName = "inventacao_car_camera_light"
        self.frames_counter = 0
        self.quaternion = None
        self.position = None

        self.qrCodeOrientationReader = None

        self.MINIMUM_ALLOWED_CONTOUR_AREA = 2000

    def onReceiveNavData(self):
        pass

    def onReceiveImuData(self, models_state):
        imuData = RosInterface.getModelState(self.modelName, models_state)
        if imuData == None:
            return
        
        pose = imuData['pose']
        twist = imuData['twist']
        #print(imuData)
        #print(pose)
        #print(type(pose))
        #print( 'position: ', pose.position)
        #print( 'orientation: ', pose.orientation)
        self.orientation = pose.orientation
        self.position = pose.position
        self.linear_velocities = twist.linear
        self.angular_velocities = twist.angular

    def cutFrame(self, frame):
        return frame[150:650,150:650]

    def processFrameAndReturnContours(self, frame):
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoothedFrame = cv2.blur(grayFrame,(5,5))
        
        # Remove outiliers pixels (too dark or too bright relative to the lane color)
        thresh = np.ones_like(smoothedFrame) * 255
        
        thresh[smoothedFrame < 45] = 0
        thresh[smoothedFrame > 80] = 0

        cv2.imshow("RobotFrontCameraOutput", thresh)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def findLaneContour(self, contours):
        biggestCountourIndex = -1
        biggerContourArea = -1
        
        for index,cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            #print("Area: ", + area)
            if area < self.MINIMUM_ALLOWED_CONTOUR_AREA:
                continue
            
            if area > biggerContourArea:
                biggerContourArea = area
                #print("Greater area: ", area)
                biggestCountourIndex = index
        
        if biggestCountourIndex >= 0:
            return contours[biggestCountourIndex]
        else:
            print("No biggest contour found")
        return np.array([])

    def getRobotOrientationAxisZ(self):
        orientation = self.orientation
        q2 = Quaternion(orientation.w, orientation.x,
            orientation.y, orientation.z)

        return q2.to_euler()[2]
    
    def interpolateTwoPoints(self, p0, p1, interpolation):
        points = [p0, p1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dt = interpolation

        x = dt * dx + p0[0]
        y = dt*dy + p0[1]
        return (x, y)

    
    def computeContourCenter (self,contour):
        M = cv2.moments(contour)
        
        contourCenterX = int(M['m10']/M['m00'])
        contourCenterY = int(M['m01'] / M['m00'])

        return (contourCenterX, contourCenterY)
    
    
    def drawLaneContourInfos(self, frame, contourCenter, rotatedBoxBoundariesCenter,
        rotatedRect, boundRect, contours_poly):

        cv2.circle(frame, contourCenter, 9, (255, 0, 0), -1)
        cv2.circle(frame, rotatedBoxBoundariesCenter[0], 20, (255, 0, 0), -1)
        cv2.circle(frame, rotatedBoxBoundariesCenter[1], 20, (0, 255, 0), -1)

        cv2.drawContours(frame, [rotatedRect], 0, (0, 0, 255), 2)
        
        # Draw contour
        cv2.drawContours(frame, [contours_poly], 0, (0,255,0))
        cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), \
        (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 0, 255), 2)
    
    def extractNextPathPointFromLaneContourAndDrawIt(self, contour, frame=np.array([])):
        contourInfos = None

        if len(contour) > 0:
            # 1- Get cotour center coordinate
            contourCenterX,contourCenterY = self.computeContourCenter(contour)
            
            # 2- Get rotated rectangle that fit the contour
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            boundRect = cv2.boundingRect(contours_poly)

            rect = cv2.minAreaRect(contour)
            #print(rect)
            box = cv2.boxPoints(rect)
            rotatedRect = np.int0(box)

            # 3- Intantiate RotatedBox class
            #   This class helps computing rotatedbox centers (opencv doesnt give it)
            rotatedObject = RotatedBox(rect)
            rotatedBoxBoundariesCenter = rotatedObject.getCenters()

            # 4- Indentify wich box center points in path's direction
            pathNextTargetPoint, _ = self.getPathNextTargetPoint(*rotatedBoxBoundariesCenter)
            
            # Instead use the inbetween point instead of the next point
            #  to maintain stability and avoid living the lane near curves
            nextTargetPointX, nextTargetPointY= self.interpolateTwoPoints(pathNextTargetPoint,(contourCenterX,contourCenterY),.5)
            
            contourInfos = {
            #'cnt': cnt,
            'centerX': contourCenterX,
            'centerY': contourCenterY,
            'pathNextTargetPoint': (nextTargetPointX, nextTargetPointY)
            }

            self.drawLaneContourInfos(frame, (contourCenterX, contourCenterY),
                rotatedBoxBoundariesCenter,rotatedRect,boundRect,contours_poly)
        else:
            print("Contour Lane is Empty")
            
        return contourInfos

    def getPathNextTargetPoint(self,center1,center2):
        # If topCenter is not know yet. Use the point more above ( the robot start going forward in it frame )
        referencePoint = (self.middleOfCamera[0], 0)
        #print('referencePoint: ', referencePoint)
        referencePoint = self.rotate_via_numpy(*referencePoint, -self.getRobotOrientationAxisZ())

        #print('rotatedReferencePoint: ', referencePoint)
        #print('robotOrientation: ', self.getRobotOrientationAxisZ())
        
        dist1 = self.computeDistance(referencePoint, center1)
        dist2 = self.computeDistance(referencePoint, center2)
        
        #print("Last topCenter: ", self.lastTopCenter)
        #print("\nCenter 1: ", center1)
        #print("Center 2: ", center2)
        if self.lastTopCenter == None:
            topCenter = None
            if dist1 < dist2:
                topCenter = center1
            else:
                topCenter = center2
            self.lastTopCenter = topCenter
        else:
            d1 = self.computeDistance(center1,self.lastTopCenter)
            d2 = self.computeDistance(center2, self.lastTopCenter)
            
            #print("center1Distance: ", d1)
            #print("center2Distance: ", d2)

            if d1 < d2:
                self.lastTopCenter = center1
            else:
                self.lastTopCenter = center2

        print("Center chosen: ", self.lastTopCenter)
        #cv2.waitKey(0)
        if (self.lastTopCenter == center1):
            #cv2.waitKey(0)
            return (self.lastTopCenter, True)
        
        return (self.lastTopCenter,False)   

    def rotate_via_numpy(self,x,y, radians):
        """Use numpy to build a rotation matrix and take the dot product."""
        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])

        return float(m.T[0]), float(m.T[1])
    
    def processBottomCameraFrame(self, frame):
        if self.orientation == None or \
            self.position == None:
                return

        if self.qrCodeOrientationReader == None:
            self.qrCodeOrientationReader = QrCodeOrientation( self.getRobotOrientationAxisZ() )


        cuttedFrame = self.cutFrame(frame)

        if self.frame_width <= 0:
            self.frame_width = cuttedFrame.shape[1]
            self.frame_height = cuttedFrame.shape[0]
            self.middleOfCamera = (int(self.frame_height / 2), int(self.frame_width / 2))

        cv2.circle(cuttedFrame, self.middleOfCamera, 15, (255, 255, 0), -1)

        contours = self.processFrameAndReturnContours(cuttedFrame)
        
        targetOrientation = self.qrCodeOrientationReader.checkAndReadQrCode(cuttedFrame)
        
        self.drawCornersMarkers(cuttedFrame)

        laneContour = self.findLaneContour(contours)

        laneContourInfos = self.extractNextPathPointFromLaneContourAndDrawIt(laneContour, cuttedFrame)
        
        if laneContourInfos != None:
            
            pathNextPoint = laneContourInfos['pathNextTargetPoint']
            pathRectangleCenter = (laneContourInfos['centerX'],laneContourInfos['centerY'])
            # Target point in camera frame
            X_c = pathNextPoint[0]
            Y_c = pathNextPoint[1]
            
            self.currentState = {
                "x": self.middleOfCamera[0],
                "y": self.middleOfCamera[1],
                "xLane": self.middleOfCamera[0],
                "yLane": self.middleOfCamera[1],
                "w": self.getRobotOrientationAxisZ()
            }

            self.PIDs.setXSetPoint(X_c)
            self.PIDs.setYSetPoint(Y_c)

            self.PIDs.setXLaneSetPoint(pathRectangleCenter[0])
            self.PIDs.setYLaneSetPoint(pathRectangleCenter[1])

            self.PIDs.setOmegaSetPoint(targetOrientation)
            print('TargetOrientation: ', targetOrientation)
            print('Robot orientation: ', self.getRobotOrientationAxisZ())

            self.controllerInputs = self.PIDs.applyPIDs(self.currentState)

            #distanceFromPathCenter = self.computeDistance(self.middleOfCamera, pathRectangleCenter)
            #distanceFromPathExtreme = self.computeDistance(self.middleOfCamera, pathNextPoint)

            #print('\nPathTarget: ', pathNextPoint)
            #print('newPathTarget: ', (X_c,Y_c))
            cv2.line(cuttedFrame, self.middleOfCamera, (int(X_c), int(Y_c)), (0, 0, 0), 5)
            #cv2.line(frame,middleOfCamera,(int(x),int(y)),(255,255,255),5)
            #cv2.waitKey(0)

            self.applyKinematics()
        
        
        self.printStateInfosOnScreen(cuttedFrame, {
            "position": self.position,
            "w": self.getRobotOrientationAxisZ(),
            "targetW": targetOrientation
        })
        cv2.imshow("RobotFrontCamera", cuttedFrame)
        
        cv2.waitKey(1)
        print('\n\n\n')

    def onImageReceived(self, frame):

        self.processBottomCameraFrame(frame)

    def printStateInfosOnScreen(self, frame, infos):
        org = (20, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        fontScale = 1
        position = infos['position']
        w = infos['w']
        targetOrientation = infos['targetW']

        text1 = 'X: ' + str(round(position.x,3))
        text2 = 'Y: ' + str(round(position.y,3))
        text3 = 'W: ' + str(round(w, 3))
        text4 = 'tW: ' + str( round(targetOrientation,3) )

        frame = cv2.putText(frame, text1, org, font, 
                   fontScale, (255,255,255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, text2, (20,100), font, 
                   fontScale, (255,255,255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, text3, (20,150), font, 
                   fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, text4, (20,200), font, 
                   fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
        

    def computeDistance(self, c1, c2):
        return ((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)**.5

   
    def applyRotationOnQuaternion(self, quaternion, angle, axis):
        r = Rotation.from_rotvec(angle * axis)
        q = r.as_quat()
        q1 = Quaternion(q[3],q[0],q[1],q[2])
        q2 = Quaternion(quaternion.w, quaternion.x,
        quaternion.y, quaternion.z)

        rotatedQuaternion = q1.quat_mult_left(q2, out='Quaternion')

        return rotatedQuaternion
    
    def computeXYVelocityInWorldFrame(self):
        xVelocity_camera = self.controllerInputs["x"]
        yVelocity_camera = self.controllerInputs["y"]

        xVelocity_camera += self.controllerInputs["xLane"]
        yVelocity_camera += self.controllerInputs["yLane"]
        
        robotOrientation = self.getRobotOrientationAxisZ()
        # Transform velocities from camera frame
        #  to robot's
        velocities_robot = self.rotate_via_numpy(xVelocity_camera, yVelocity_camera, robotOrientation)
        xVelocity_robot, yVelocity_robot = velocities_robot

        """ World       Robot
            -> Y        -> X        
            |           |             
            v           v              
            X           Y              

            World -> Robot
            Swap X and Y coordinates or apply the rotations
             Z(-90)X(-180) to the point in robot frame
        """
        xVelocity_world = yVelocity_robot
        yVelocity_world = xVelocity_robot

        return (xVelocity_world, yVelocity_world)

    
    def applyKinematicsInPosition(self):
        from geometry_msgs.msg import Point

        dt = time() - self.last_time

        xVelocity_world, yVelocity_world = \
             self.computeXYVelocityInWorldFrame()

        newPosition = Point()

        newX = self.position.x + xVelocity_world * dt
        newY = self.position.y + yVelocity_world * dt

        newPosition.x =  newX
        newPosition.y = newY
        newPosition.z = 0.361203

        return newPosition

    def applyKinematicsInOrientation(self):
        dt = time() - self.last_time

        wVelocity = self.controllerInputs["w"]
        
        wAngle = wVelocity * dt

        newOrientation = self.applyRotationOnQuaternion(self.orientation, wAngle, np.array([0, 0, 1]))

        return newOrientation
        
    def applyKinematics(self):
        
        newPosition = self.applyKinematicsInPosition()

        newOrientation = self.applyKinematicsInOrientation()

        objstate = self.objstate
        objstate.model_state.model_name = self.modelName
        objstate.model_state.pose.position = newPosition
        objstate.model_state.pose.orientation = newOrientation
        objstate.model_state.reference_frame = "world"
        
        result = self.set_state_service(objstate)

        self.last_time = time()


    def updateBottomLigtherPosition(self):
        set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #self.objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        # set red cube pose
        objstate = SetModelStateRequest()
        objstate.model_state.model_name = self.bottomLightName
        objstate.model_state.pose.position.x = self.position.x
        objstate.model_state.pose.position.y = self.position.y
        #objstate.model_state.pose.position.z = inventacaoCarPose.position.z
        #objstate.model_state.pose.orientation.w = self.orientation.w
        #objstate.model_state.pose.orientation.x = self.orientation.x
        #objstate.model_state.pose.orientation.y = self.orientation.y
        #objstate.model_state.pose.orientation.z = self.orientation.z
        
        objstate.model_state.reference_frame = "world"
        
        result = set_state_service(objstate)

def onRosShutdown():
    print("Ros is being shutdown")
    cv2.destroyAllWindows()

def onImageReceived(frame):
    cv2.imshow("Person1View", frame)
    cv2.waitKey(1)

def onReceiveModelsState(models_state):
    car_state = RosInterface.getModelState("inventacao_car",models_state)
    #print(car_state)

if __name__ == "__main__":
    
    rosInterface = RosInterface('InventacaoCar')
    car = InventacaoCarCameraBelow('inventacao_car')
    rospy.wait_for_service('/gazebo/set_model_state')
    rospy.on_shutdown(onRosShutdown)
    
    rosInterface.listenToCamera("/inventacao_car/camera1/image_raw", car.onImageReceived)
    rosInterface.listenToModelsStates(car.onReceiveImuData)

    #gazeboCameraPerson1 = GazeboCamera("/inventacao_car/cameraPerson1/image_raw")
    #gazeboCameraPerson1.setOnReceiveImage(onImageReceived)

    print('1')
    #car.moveRobot(0, 0)
    print('2')
    
    #sub_model_states = rospy.Subscriber("/gazebo/model_states", gazeboMsgs.ModelStates, onReceiveModelsState)
    #robotCamera = GazeboCamera("/inventacao_car/camera1/image_raw")
    #robotCamera.setOnReceiveImage(onImageReceived)

    while rospy.is_shutdown() == False:
        print('3')
        rospy.spin()

    rospy.signal_shutdown( "Fim" )
    print('teste')
