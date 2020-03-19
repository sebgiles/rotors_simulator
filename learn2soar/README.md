roslaunch rotors_gazebo fixed_wing_hil.launch verbose:=true uav_name:=soaring

rostopic pub -r 100 /uav_1/prop_ref_0 std_msgs/Float32 -- 100000

rostopic pub -1 /uav_1/elev_pos std_msgs/Float32 -- 100

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: 'uav_1', pose: {position: {x: 10, y: 0, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 1}}, twist: {linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}, reference_frame: ground_collision_plane}"
