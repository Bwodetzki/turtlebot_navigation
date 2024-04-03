import time
import numpy as np
import pybullet as p

physics_server_id = None

turtle_offset = [0,0,0]
turtle = None

lidar = None

class Lidar():
    def __init__(self, lidarDist, lidarAngle, numMeasurements):
        self.dist = lidarDist
        self.spreadAngle = lidarAngle
        self.numMeasurements = numMeasurements

def disconnect():
    global physics_server_id
    if connected():
        p.disconnect()
        physics_server_id = None

def connected():
    global physics_server_id
    if physics_server_id is None or not p.isConnected(physics_server_id):
        physics_server_id = None # needed in case sim is closed with button and server id isn't updated
        return False
    return True

def create_sim(gui=True, load_turtle=True, turtle_opts={'pos':[0,0,0], 'orn':[0,0,0,1]}):
    """Create and connect to a physics server, optionally with a GUI visualizer

    Args:
        gui (bool, optional): Loads the simulation with the debugging GUI. Defaults to True.
        load_turtle (bool, optional): Loads the simulation with the turtle. Defaults to True.
    """
    global turtle
    global physics_server_id
    if not connected():
        if gui:
            physics_server_id = p.connect(p.GUI)
        else:
            physics_server_id = p.connect(p.DIRECT)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_KEYBOARD_SHORTCUTS, enable=0)

        plane = p.loadURDF("plane.urdf")
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,-10)
        if load_turtle:
            turtle = p.loadURDF("turtlebot.urdf",turtle_opts['pos'],turtle_opts['orn'])
            return turtle

def clear_sim():
    p.resetSimulation()
    turtle = None

def create_box(position, dimensions, angles=[0, 0, 0], mass=0):
    quat = p.getQuaternionFromEuler(angles)
    obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dim/2 for dim in dimensions])
    p.createMultiBody(mass, obstacle, basePosition=position, baseOrientation=quat)

def create_waypoint(position, radius, color=[1, 0, 0, 1]):
    waypoint = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    p.createMultiBody(baseMass=0,
                      basePosition=position,
                      baseVisualShapeIndex=waypoint)

def initLidar(lidarDist: float = 1, lidarAngle: float = 2*np.pi, numMeasurements: int = 360):
    global lidar
    lidar = Lidar(lidarDist, lidarAngle, numMeasurements)


def localLidar(botPos: tuple[float, float, float], botAngle: float, draw: bool = False, eraseOld: bool = True):
    """Calculate and optionally draw lidar measurements given a position and lidar parameters.

    Args:
        botPos (tuple[float, float, float]): X, Y, and Z coordinates of the bot (actual lidar is run above the bot).
        botAngle (float): Head angle of the bot.
        lidarDist (float, optional): Distance each lidar sensor can detect up to. Defaults to 1.
        lidarAngle (float, optional): Spread of lidar sensor angles on the bot. Centered around the bot's head angle. Defaults to 2*np.pi.
        numMeasurements (int, optional): Number of lidar sensors. Defaults to 360.
        draw (bool, optional): Draws visual objects showing lidar sensors' range and hit fractions. Defaults to False.
        eraseOld (bool, optional): When draw is True, erases old visual objects when drawing new ones. Defaults to True.

    Returns:
        list[float]: list of hit fractions between 0 and 1
    """
    global lidar
    visualRayWidth = 0.01 # width of visual rays drawn when "draw" is True
    visualRayNonCollisionColor = [0, 1, 0] # RGB from 0-1 for visual rays until collision point
    visualRayCollisionColor = [1, 0, 0] # RGB from 0-1 for visual rays after collision point
    fullVisual = False # shows full extent of lidar lines, even through obstacles
    sensor_height = 1 # taller than the bot so rays won't intersect with its body
    bot_radius = .17 # just smaller than the bot radius to stop rays from starting inside a very close obstacle
    relStartAngle = -lidar.spreadAngle/2 # relative start angle to bot heading angle
    worldStartAngle = botAngle + relStartAngle
    if lidar.numMeasurements > 1:
        angleStep = lidar.spreadAngle/(lidar.numMeasurements-1)
    else:
        angleStep = 0
    measurementAngles = [worldStartAngle + angleIdx*(angleStep) for angleIdx in range(lidar.numMeasurements)]
    rayStartPoints = [np.array(botPos) + bot_radius*np.array([np.cos(angle), np.sin(angle), sensor_height]) for angle in measurementAngles]
    rayEndPoints = [startPoint + lidar.dist*np.array([np.cos(angle), np.sin(angle), 0]) for startPoint, angle in zip(rayStartPoints, measurementAngles)]
    rayInfoArray = p.rayTestBatch(rayStartPoints, rayEndPoints)
    measurements = [hit_fraction for _, _, hit_fraction, _, _ in rayInfoArray]
    if draw:
        if eraseOld:
            if len(localLidar.oldVisuals) > 0: # the first time this is called, this array will be empty
                for idx, (rayStart, rayEnd, measurement) in enumerate(zip(rayStartPoints, rayEndPoints, measurements)):
                    collisionPoint = rayStart + measurement * (rayEnd - rayStart)
                    if fullVisual:
                        idx1 = 2*idx
                        idx2 = 2*idx + 1
                    else:
                        idx1 = idx
                        idx2 = None # value should never be used
                    localLidar.oldVisuals[idx1] = p.addUserDebugLine(rayStart, collisionPoint, visualRayNonCollisionColor, visualRayWidth, 0, replaceItemUniqueId=localLidar.oldVisuals[idx1])
                    if fullVisual:
                        localLidar.oldVisuals[idx2] = p.addUserDebugLine(collisionPoint, rayEnd, visualRayCollisionColor, visualRayWidth, 0, replaceItemUniqueId=localLidar.oldVisuals[idx2])
            else:
                for rayStart, rayEnd, measurement in zip(rayStartPoints, rayEndPoints, measurements):
                    collisionPoint = rayStart + measurement * (rayEnd - rayStart)
                    localLidar.oldVisuals.append(p.addUserDebugLine(rayStart, collisionPoint, visualRayNonCollisionColor, visualRayWidth, 0))
                    if fullVisual:
                        localLidar.oldVisuals.append(p.addUserDebugLine(collisionPoint, rayEnd, visualRayCollisionColor, visualRayWidth, 0))
        else:
            for rayStart, rayEnd, measurement in zip(rayStartPoints, rayEndPoints, measurements):
                collisionPoint = rayStart + measurement * (rayEnd - rayStart)
                p.addUserDebugLine(rayStart, collisionPoint, visualRayNonCollisionColor, visualRayWidth, 0)
                if fullVisual:
                    p.addUserDebugLine(collisionPoint, rayEnd, visualRayCollisionColor, visualRayWidth, 0)
    return measurements
localLidar.oldVisuals = []

def step_sim(leftWheelVelocity, rightWheelVelocity, visualizeLidar=False):
    try: # needed if simulation stops suddenly
        p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
        p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)
        if visualizeLidar:
            turtle_pos, turtle_orientation = getTurtleInfo()
            turtle_orientation = p.getEulerFromQuaternion(turtle_orientation)[-1]
            localLidar(turtle_pos, turtle_orientation, draw=True, eraseOld=True)
    except Exception:
        return

def keyboard_control(forward, turn, speed=10, camStep=0.1, camAngleStep=1):
    try: # needed if simulation stops suddenly
        global physics_server_id
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
            p.disconnect() # Quit simulator if uppercase Q is pressed
            physics_server_id = None
            return (leftWheelVelocity, rightWheelVelocity, forward, turn)
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
    except Exception:
        physics_server_id = None
    finally:
        return (leftWheelVelocity, rightWheelVelocity, forward, turn)

def getTurtleInfo():
    '''
    Returned value in the form of (turtle_position, turtle_orientation_quaternion)
    '''
    return p.getBasePositionAndOrientation(turtle)

def main():
    import env_manager as em
    boundaryFile = './envData/env0/env_boundary.dat'
    obstaclesFile = './envData/env0/env_obstacles.dat'
    create_sim()
    em.load_env(boundaryFile, obstaclesFile)
    create_box([3,3,1.5], dimensions=[3,3,3], angles=[0,0,0], mass=0)
    create_box([-3,3,1.5], dimensions=[3,3,3], angles=[0,0,0], mass=0)
    create_box([3,-3,1.5], dimensions=[3,3,3], angles=[0,0,0], mass=0)
    create_box([-3,-3,1.5], dimensions=[3,3,3], angles=[0,0,0], mass=0)
    create_waypoint([1, 0, 0.25], radius=1/8)
    initLidar(lidarDist=10, lidarAngle=2*np.pi, numMeasurements=36)
    forward=0
    turn=0
    startTime = time.time()
    samplePeriod = 5 # seconds
    while connected(): # use sim.connected() instead of plt.isConnected so we don't need a server id
        time.sleep(1./240.)
        leftWheelVelocity, rightWheelVelocity, forward, turn = keyboard_control(forward, turn, speed=30)
        step_sim(leftWheelVelocity, rightWheelVelocity, visualizeLidar=True)
        
        currTime = time.time()
        if currTime > startTime + samplePeriod:
            # turtle_pos, turtle_orientation = getTurtleInfo()
            # turtle_orientation = p.getEulerFromQuaternion(turtle_orientation)[-1]
            # lidarResults = localLidar(turtle_pos, turtle_orientation, draw=True, eraseOld=True)
            # print(f'{turtle_orientation = }')
            # print(f'{lidarResults = }')
            startTime = currTime
    disconnect()

if __name__ == '__main__':
    main()