import pybullet as p
import numpy as np
import time
import simulation as sim

def angle_diff(theta1, theta2):
    diff = theta2 - theta1
    return np.arctan2(np.sin(diff), np.cos(diff))

def controller_v1(target_pos, turtlebot_id, max_vel=10, eps=1e-1):
    ''' 
    Velocity controller for turtlebot.

    Inputs:
    target_pos: Target Position coordinate
        np.array of length 3

    turtlebot_id: Id of turtlebot object
        idk
    
    Main Issues:
    2 modes: turning and drive straight. Not very natural. (when eps is small, the turtlebot will overshoot the turning and have jittery behavior)
    '''

    # Note that orn = orientation
    curr_pos, curr_orn = p.getBasePositionAndOrientation(turtlebot_id)
    curr_orn = p.getEulerFromQuaternion(curr_orn)

    # Lets move to 2d
    curr_orn = np.array(curr_orn[-1])
    curr_pos = np.array(curr_pos[0:2])
    target_pos = np.array(target_pos[0:2])

    desired_traj = target_pos - curr_pos
    desired_orn = np.arctan2(desired_traj[1], desired_traj[0])

    # Check if at target
    if np.linalg.norm(desired_traj) < eps:
        return (0, 0)

    # Calculate Wheel Velocities
    leftWheelVelocity=0
    rightWheelVelocity=0

    orn_error = angle_diff(curr_orn, desired_orn)
    if abs(orn_error)>eps:
        leftWheelVelocity=-1*max_vel*np.sign(orn_error)
        rightWheelVelocity=-1*leftWheelVelocity
    else:
        leftWheelVelocity=max_vel
        rightWheelVelocity=max_vel
    
    return (leftWheelVelocity, rightWheelVelocity)


def controller_v2(target_pos, turtlebot_id, a=5, max_vel=10, eps=1e-2):
    ''' 
    Velocity controller for turtlebot with more natural movement.
    Key idea: instead of a hard cutoff at epsilon when turning and going straight, one should use a smooth function to transtition between the two. An exponential is a reasonable choice. 

    Notes that "a" is not well tuned right now
    The higher the "a", the faster it will want to turn (above 5-10 the difference is minimal)

    The lower the "a" the more it will want to go forward

    Maximum velocity should be determined on a wheel by wheel basis (since at least in real life, wheel slippage is the limmiting factor)
    Some velocity normalization would solve this.

    Inputs:
    target_pos: Target Position coordinate
        np.array of length 3

    turtlebot_id: Id of turtlebot object
        idk
    '''

    func = lambda x: np.exp(-a*x)
    # Note that orn = orientation
    curr_pos, curr_orn = p.getBasePositionAndOrientation(turtlebot_id)
    curr_orn = p.getEulerFromQuaternion(curr_orn)

    # Lets move to 2d
    curr_orn = np.array(curr_orn[-1])
    curr_pos = np.array(curr_pos[0:2])
    target_pos = np.array(target_pos[0:2])

    desired_traj = target_pos - curr_pos
    desired_orn = np.arctan2(desired_traj[1], desired_traj[0])

    # Check if at target
    if np.linalg.norm(desired_traj) < eps:
        return (0, 0)

    # Calculate Wheel Velocities
    leftWheelVelocity=0
    rightWheelVelocity=0

    orn_error = angle_diff(curr_orn, desired_orn)
    
    leftWheelVelocity = max_vel*(func(abs(orn_error)) - (1 - func(abs(orn_error)))*np.sign(orn_error))
    rightWheelVelocity = max_vel*(func(abs(orn_error)) + (1 - func(abs(orn_error)))*np.sign(orn_error))
    
    return (leftWheelVelocity, rightWheelVelocity)


if __name__=='__main__':
    sim.create_sim()

    target = np.array([-5, 1, 0])  # Set the target for the turtlebot here

    while (1):
        time.sleep(1./240.)
        keys = p.getKeyboardEvents()
        leftWheelVelocity=0
        rightWheelVelocity=0
        speed=10

        # Find Wheel Velocities
        leftWheelVelocity, rightWheelVelocity = controller_v2(target, sim.turtle, max_vel=speed)

        # Update Turtle Bot
        sim.step_sim(leftWheelVelocity, rightWheelVelocity)