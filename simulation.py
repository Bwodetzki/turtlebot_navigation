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

def create_box(position, dimensions, mass=0):
    obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dim/2 for dim in dimensions])
    p.createMultiBody(mass, obstacle, basePosition=position)

def step_sim(leftWheelVelocity, rightWheelVelocity):
    p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
    p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

def load_env(bulletFile):
    p.restoreState(fileName=bulletFile)