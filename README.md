# turtlebot_navigation
Navigation algorithms for turtlebot using pybullet. RRT* and NN based navigation.

## Quick Start
Python 3.11.6 is used. \
Use requirements.txt for version control. \

## Demos of Files
Running turtle_path_following.py with the default parameters will generate a random environment and a random path and track the generated RRT* path. To use a trained model, set --run 5 to be the number 5. Note: the path generation takes a long time. 

## Useful Resources
The turtlbot URDF and orginal test code was taken from this repo: https://github.com/erwincoumans/pybullet_robots?tab=readme-ov-file

The orginal pybullet repo is also helpful for understanding how pybullet works (the link is the quickstart guide): https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html

This more in depth guide is helpful in understanding the basics: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit