#!/bin/sh
X=-50
Y=0
Z=30
VX=15
VY=-15
rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: 'uav_1', pose: {position: {x: $X, y: $Y, z: $Z}, orientation: {x: 0, y: 0, z: 0, w: 0}}, twist: {linear: {x: $VX, y: $VY, z: 0}, angular: {x: 0, y: 0, z: 0}}, reference_frame: ground_collision_plane}"
