Goal: Test files which will allow us to take these measurements.

Measurements
Things to track while running :
CPU usage
GPU usage 
Time to compute
memory usage
power usage? 
(can optionally do this through wandb board, I have seen this being used for ML training)

Navigation (not as important)
Success rate: Number of times reached the exit
Collisions: How many times collided with an object (path blocked)
Path length: How long the path length was to the exit
Average successful path length

Perception
We can make both an occupancy grid and a point cloud so we can evaluate both against ground truth
2D or 3D occupancy grid: Use occupancy IoU (intersection over union) for predicting occupied vs empty spaces
Point cloud chamfer distance measurements for each point in the point cloud
RANSAC inlier counts for classical
Translation and rotation error like from Homework 1
Can evaluate against ground truth through manually finding the shortest distance to the exit, OR, having a human explore and tracking that path. 
The first option is easier. 
Evaluate against ground truth after first evaluating against random actions. 
