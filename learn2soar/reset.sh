#!/bin/sh

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: 'uav_1', pose: {position: {x: 0, y: -100, z: 50}, orientation: {x: 0, y: 0, z: 0, w: 1}}, twist: {linear: {x: 20, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}, reference_frame: ground_collision_plane}"
