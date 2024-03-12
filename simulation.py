import pybullet as p
turtle = None

def create_sim():
    global turtle
    if turtle is None:
        p.connect(p.GUI)
        turtle_offset = [0,0,0]
        turtle = p.loadURDF("turtlebot.urdf",turtle_offset)
        plane = p.loadURDF("plane.urdf")
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,-10)

def load_sim(bulletFile):
    global turtle
    if turtle is None:
        p.connect(p.GUI)
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

def keyboard_control(forward, turn):
    keys = p.getKeyboardEvents()
    leftWheelVelocity=0
    rightWheelVelocity=0
    speed=10

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

    rightWheelVelocity+= (forward+turn)*speed
    leftWheelVelocity += (forward-turn)*speed

    return (leftWheelVelocity, rightWheelVelocity, forward, turn)