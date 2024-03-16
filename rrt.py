"""
Path Planning Sample Code with RRT*

author: Zach Cawood and Brian Wodetzki, code adapted from Ahmed Qureshi

"""

import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import time
import collision_checker as cc
import env_manager as em


def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2))

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, boundary, obstacles, sampleArea, turtle_radius, alg, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=50):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start)
        self.end = Node(goal)
        self.obstacles = obstacles
        self.boundary = boundary
        self.sampleArea = sampleArea
        self.turtle_radius = turtle_radius
        self.alg = alg
        self.dof = dof

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.goalfound = False
        self.solutionSet = set()

    def planning(self, animation=False):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        break_flag=False
        counter = 0  # Counts how many iters have passed since a solution has been found, breaks after 5!!

        for i in range(self.maxIter):
            print(i)
            rnd = self.generatesample()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            if i==0:
                rnd_valid, rnd_cost = self.steerTo(self.end, self.start)
                if rnd_valid:
                    rnd = self.end
                    if not animation:
                        break_flag=True
                else:
                    rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])
            else:
                rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])

            if self.goalfound == True:
                    counter+=1
                    if counter>=5:
                        break_flag=True

            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar':
                    self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

                if self.is_near_goal(newNode):
                    is_valid_sol, cost = self.steerTo(self.end, newNode)
                    if is_valid_sol or cost==None:
                        self.solutionSet.add(newNodeIndex)
                        self.goalfound = True


                if animation:
                    self.draw_graph(rnd.state)
                
                if break_flag:
                    break

        return self.get_path_to_goal()

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        costs = [dist(newNode.state, self.nodeList[parent].state) + self.nodeList[parent].cost for parent in nearinds]
        
        sorted_idxs = [x for _, x in sorted(zip(costs, nearinds))]

        parent = None
        for idx in sorted_idxs:
            success, _ = self.steerTo(newNode, self.nodeList[idx])
            if success:
                parent = idx
                break

        return parent

    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        This function is used in planning() to filter out invalid random samples. You may also find it useful
        for implementing the functions in question 1.

        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)
        for j in range(0,self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)


        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0,self.dof):
                dists[j] =dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1

            # stateCurr = np.zeros(self.dof,dtype=np.float32)
            # for j in range(0,self.dof):
            #     stateCurr[j] = newNode.state[j]
            stateCurr = newNode.state

            stateCurr = Node(stateCurr)

            for i in range(0,numSegments):

                if not self.__CollisionCheck(stateCurr):
                    return (False, None)

                for j in range(0,self.dof):
                    stateCurr.state[j] += dists[j]

            if not self.__CollisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def generatesample(self, max_iters=100):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """
        if random.randint(0, 100) > self.goalSampleRate:
            iters = 0
            while iters < max_iters:
                # Generate Sample
                sample = (2*np.random.rand(2)-1)*self.sampleArea

                # Check Conditions for Start
                rect_cols = [cc.rectangle_col_checker(state, sample, self.turtle_radius) for state in self.obstacles]
                condition =  (not any(rect_cols)) and cc.is_inside_boundary(self.boundary, sample, self.turtle_radius)
                if condition:
                        break
                iters+=1
            assert (iters != max_iters)  # Did not converge to the start position in the given # of iterations
            rnd = Node(sample)
        else:
            rnd = self.end

        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = dist(node.state, self.end.state)
        if d < 5.0:
            return True
        return False

    @staticmethod
    def get_path_len(path):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += dist(path[i], path[i-1])

        return pathLen


    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.end.state]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append(node.state)
            goalind = node.parent
        path.append(self.start.state)
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        # Use this value of gamma
        GAMMA = 50
        v = len(self.nodeList)
        condition = GAMMA*(np.log(v)/v)**(1/self.dof)

        neighbors = []
        for i, node in enumerate(self.nodeList):
            distance = dist(newNode.state, node.state)
            if distance <= condition:
                neighbors.append(i)

        # your code here
        return neighbors

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
        Recall that a node should be rewired if doing so would reduce its cost.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        # your code here
        # if nearinds:
            # nearinds.remove(newNode.parent)  # shouldnt matter cost will be less
        curr_costs = [self.nodeList[node].cost for node in nearinds]
        modified_costs = [dist(newNode.state, self.nodeList[node].state) + newNode.cost for node in nearinds]
        for i in range(len(nearinds)):
            condition, _ = self.steerTo(newNode, self.nodeList[nearinds[i]])
            if (modified_costs[i] < curr_costs[i]) & condition:
                newNode.children.add(nearinds[i])
                self.nodeList[self.nodeList[nearinds[i]].parent].children.remove(nearinds[i])
                self.nodeList[nearinds[i]].parent = newNodeIndex
                self.nodeList[nearinds[i]].cost = modified_costs[i]
                self.nodeList[nearinds[i]].update_childrens_cost(self.nodeList)
            else:
                pass


    def GetNearestListIndex(self, nodeList, rnd):
        """
        Searches nodeList for the closest vertex to rnd

        nodeList: list of all nodes currently in the tree
        rnd: node to be added (not currently in the tree)

        Returns: index of nearest node
        """
        dlist = []
        for node in nodeList:
            dlist.append(dist(rnd.state, node.state))

        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node):
        """
        Checks whether a given configuration is valid. (collides with obstacles)

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        point = node.state
        rect_cols = [cc.rectangle_col_checker(state, point, self.turtle_radius) for state in self.obstacles]
        is_valid =  (not any(rect_cols)) and cc.is_inside_boundary(self.boundary, point, self.turtle_radius)

        return is_valid  # safe'''

    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        if self.goalfound:
            goalind = None
            mincost = float('inf')
            for idx in self.solutionSet:
                cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    def draw_graph(self, rnd=None):
        """
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # This is our barrier
        plt.gca().plot(self.boundary[:, 0], self.boundary[:, 1], 'b')
        plt.gca().plot([self.boundary[0, 0], self.boundary[-1, 0]], [self.boundary[0, 1], self.boundary[-1, 1]], 'b')

        # These are our obstacles
        for obstacle in self.obstacles:
            em.plot_obstacle(obstacle, ax=plt.gca())


        for node in self.nodeList:
            if node.parent is not None:
                if node.state is not None:
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
                        node.state[1], self.nodeList[node.parent].state[1]], "-g")
                    circle = mpatches.Circle((node.state[0], node.state[1]), self.turtle_radius, color='b')
                    plt.gca().add_patch(circle)
                    # elif self.geom == 'rectangle':
                    #     rect = mpatches.Rectangle((node.state[0]-3/2, node.state[1]-3/4), 3, 3/2, fill=True, color="b", linewidth=0.1)
                    #     tr = mpl.transforms.Affine2D().rotate_deg_around(node.state[0], node.state[1], np.rad2deg(node.state[2])) + plt.gca().transData
                    #     rect.set_transform(tr)
                    #     plt.gca().add_patch(rect)


        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, '-r')

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")

        plt.plot(self.start.state[0], self.start.state[1], "xr")
        plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.title("Path of the turtlebot for the " + self.alg + " Algorithm")
        plt.grid(True)
        plt.pause(0.1)



class Node():
    """
    RRT Node
    """

    def __init__(self,state):
        self.state = state
        self.cost = 0.0
        self.parent = None
        self.children = set()
    
    def update_childrens_cost(self, nodeList):
        for child in self.children:
            distance = np.linalg.norm(np.array(self.state) - np.array(nodeList[child].state))
            nodeList[child].cost = self.cost + distance
            nodeList[child].update_childrens_cost(nodeList)

def rrt_star(boundary, obstacles, start, goal, RRTs_params):
    
    rrt = RRT(start=start, 
              goal=goal, 
              boundary=boundary, 
              obstacles=obstacles, 
              sampleArea=RRTs_params['sample_bounds'], 
              turtle_radius=RRTs_params['turtle_radius'], 
              alg='rrtstar', 
              dof=2, 
              maxIter=RRTs_params['max_iters'])
    path = rrt.planning(animation=False)
    return path
    


def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='circle', choices=['point', 'circle', 'rectangle'], \
        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrtstar', choices=['rrt', 'rrtstar'], \
        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=50, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')

    args = parser.parse_args()

    show_animation = not args.blind and not args.fast
    # show_animation = False

    print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
    starttime = time.time()

    dof=2
    barrier_seed = None
    obstacle_seed = None
    sg_seed = None

    center = (0,0)
    avg_radius = 10
    irregularity = 1.0
    spikiness = 0.4
    num_vertices = 10
    min_radius = 1
    barrier_vertices = em.generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices, min_radius, barrier_seed)

    center_bounds = [20,20]
    edge_len_bounds = [0.1, 2]
    num_obstacles = 40
    max_iters = 1000
    obstacles = em.generate_obstacles(barrier_vertices, center_bounds, edge_len_bounds, obstacle_seed, num_obstacles, max_iters)

    radius = 0.75
    center_bounds = np.array([20, 20])
    min_dist_from_start_to_goal = 1
    start, goal, angle = em.generate_start_goal(barrier_vertices, obstacles, radius, center_bounds, min_dist_from_start_to_goal, sg_seed)

    rrt = RRT(start=start, goal=goal, boundary=barrier_vertices, obstacles=obstacles, sampleArea=center_bounds, turtle_radius=0.1, alg=args.alg, dof=dof, maxIter=args.iter)
    path = rrt.planning(animation=show_animation)

    endtime = time.time()

    if path is None:
        print("FAILED to find a path in %.2fsec"%(endtime - starttime))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec"%(RRT.get_path_len(path), endtime - starttime))
    # Draw final path
    if not args.blind:
        rrt.draw_graph()
        plt.show()

    if path is None:
        return np.array((np.NaN, (endtime - starttime), 0))
    else:
        return np.array((RRT.get_path_len(path), (endtime - starttime), 1))
    


if __name__ == '__main__':
    main()