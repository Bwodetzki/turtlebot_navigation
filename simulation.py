import numpy as np
import pybullet as p
turtle_offset = [0,0,0]
turtle = None

def create_sim():
    global turtle
    if turtle is None:
        p.connect(p.GUI)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_KEYBOARD_SHORTCUTS, enable=0)
        turtle = p.loadURDF("turtlebot.urdf",turtle_offset)
        plane = p.loadURDF("plane.urdf")
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,-10)

def load_sim(bulletFile):
    global turtle
    if turtle is None:
        p.connect(p.GUI)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_KEYBOARD_SHORTCUTS, enable=0)
        load_env(bulletFile)
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,-10)

def create_box(position, dimensions, angles=[0, 0, 0], mass=0):
    quat = p.getQuaternionFromEuler(angles)
    obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dim/2 for dim in dimensions])
    p.createMultiBody(mass, obstacle, basePosition=position, baseOrientation=quat)

def step_sim(leftWheelVelocity, rightWheelVelocity):
    p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
    p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

def load_env(bulletFile):
    p.restoreState(fileName=bulletFile)

def keyboard_control(forward, turn, speed=10, camStep=0.1, camAngleStep=1):
    leftWheelVelocity=0
    rightWheelVelocity=0
    freeCam = False
    camStep = 0.1
    camInfo = p.getDebugVisualizerCamera()
    camUpAxis = camInfo[-8] # vertical world axis
    camUpAxis /= np.sqrt(np.dot(camUpAxis, camUpAxis))
    camForwardAxis = camInfo[-7] # axis extending from front of camera
    camForwardAxis /= np.sqrt(np.dot(camForwardAxis, camForwardAxis))
    camHorizontalAxis = camInfo[-6] # axis extending from side of camera
    camHorizontalAxis /= np.sqrt(np.dot(camHorizontalAxis, camHorizontalAxis))
    camVerticalAxis = camInfo[-5] # axis extending from top of camera
    camVerticalAxis /= np.sqrt(np.dot(camVerticalAxis, camVerticalAxis))
    camFrontAxis = camForwardAxis # flattened version of forward axis
    camFrontAxis[2] = 0
    camFrontAxis /= np.sqrt(np.dot(camFrontAxis, camFrontAxis))
    camYaw = camInfo[-4]
    camPitch = camInfo[-3]
    camDistToTarget = camInfo[-2]
    camTargetPos = camInfo[-1]

    keys = p.getKeyboardEvents()
    if p.B3G_SHIFT in keys and ord('q') in keys:
        quit() # Quit simulator if uppercase Q is pressed
    if p.B3G_SHIFT in keys and ord('r') in keys:
        p.resetBasePositionAndOrientation(bodyUniqueId=turtle, posObj=turtle_offset, ornObj=[0,0,0,1]) # reset turtlebot
    for k,v in keys.items():
        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
            turn = -0.5
        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
            turn = 0
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
            turn = 0.5
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
            turn = 0

        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
            forward=1
        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
            forward=0
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
            forward=-1
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
            forward=0      

        # TODO: make camera follow wherever mouse is pointing
        # if (k == ord('c') and (v&p.KEY_WAS_TRIGGERED)):
        #     freeCam = True
        # if (k == ord('c') and (v&p.KEY_WAS_RELEASED)):
        #     freeCam = False

        if (k == ord('w') and (v&p.KEY_IS_DOWN)):
            camTargetPos += camStep*camFrontAxis # move camera forward
        if (k == ord('s') and (v&p.KEY_IS_DOWN)):
            camTargetPos -= camStep*camFrontAxis # move camera back
        if (k == ord('a') and (v&p.KEY_IS_DOWN)):
            camTargetPos -= camStep*camHorizontalAxis # move camera left
        if (k == ord('d') and (v&p.KEY_IS_DOWN)):
            camTargetPos += camStep*camHorizontalAxis # move camera right
        if (k == ord('f') and (v&p.KEY_IS_DOWN)):
            camTargetPos -= camStep*camUpAxis # move camera down
        if (k == ord('r') and (v&p.KEY_IS_DOWN) and p.B3G_SHIFT not in keys):
            camTargetPos += camStep*camUpAxis # move camera up
        if (k == ord('q') and (v&p.KEY_IS_DOWN)):
            camYaw += camAngleStep # turn camera left
        if (k == ord('e') and (v&p.KEY_IS_DOWN)):
            camYaw -= camAngleStep # turn camera right
        if (k == ord('z') and (v&p.KEY_IS_DOWN)):
            camPitch += camAngleStep # tilt camera up
        if (k == ord('x') and (v&p.KEY_IS_DOWN)):
            camPitch -= camAngleStep # tilt camera down


    for event in p.getMouseEvents():
        if freeCam and event[0] == 1:
            pass # TODO: make camera follow wherever mouse is pointing

    rightWheelVelocity += (forward+turn)*speed
    leftWheelVelocity += (forward-turn)*speed

    p.resetDebugVisualizerCamera(camDistToTarget, cameraYaw=camYaw, cameraPitch=camPitch, cameraTargetPosition=camTargetPos)

    return (leftWheelVelocity, rightWheelVelocity, forward, turn)