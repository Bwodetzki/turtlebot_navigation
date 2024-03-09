# turtlebot_navigation
Navigation algorithms for turtlebot using pybullet. RRT* and NN based navigation.

## Quick Start
Python 3.11.6 is used. \
Use requirements.txt for version control. \
turtlebot_demo is a neat demo of the turtlebot (go figure)

## TODO:
1. Decide what environment we want to use and implement the environment in pybullet. 
    - I see two options for a base environment for this class, one is a bunch of randomly generated rectangles as obstacles, and the other is a bunch of randomly generated tables and chairs and other more specific items. Both provide trainable environment, the former would be easier, the latter would allow us to develop and test the algorithm for a more real worls environment (I think of a roomba)
2. Develop RRT* algorithm
3. Develop simple controller so bot can track trajectory
4. Write midterm report (Due 3/18)
    - Am I missing something?
5. Implement autonomy
6. Final report (Due 4/11)



## Useful Resources
The turtlbot URDF and orginal test code was taken from this repo: https://github.com/erwincoumans/pybullet_robots?tab=readme-ov-file

The orginal pybullet repo is also helpful for understanding how pybullet works (the link is the quickstart guide): https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html

This more in depth guide is helpful in understanding the basics: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit

We should find some papers that go over algorithms similar to what we want to replicate in this project.