#!/bin/sh

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: 'uav_1', pose: {position: {x: 10, y: -50, z: 45}, orientation: {x: 0, y: 0, z: 0, w: 1}}, twist: {linear: {x: 25, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}, reference_frame: ground_collision_plane}"
