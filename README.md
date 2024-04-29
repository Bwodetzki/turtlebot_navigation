# turtlebot_navigation
Navigation algorithms for turtlebot using pybullet. RRT* and NN based navigation.

## Quick Start
Python 3.11.6 is used. \
Use requirements.txt for version control.

### Visualize trained models
To visualize a trained model, identify the type of model you wish to run (MPNet or RNN) and find the corresponding run # in the good_models/ or RNN_good_models/complicatedModels/ folders. Then, to visualize the model working, run "python turtle_path_following.py --run <run#> --env <env#> --path <path#>" with the "--rnn" flag if running the RNN model.

## Useful Resources
The turtlebot URDF and original test code was taken from this repo: https://github.com/erwincoumans/pybullet_robots?tab=readme-ov-file

The original pybullet repo is also helpful for understanding how pybullet works (the link is the quickstart guide): https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html

This more in depth guide is helpful in understanding the basics: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit