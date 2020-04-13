# to get going
- You need 4 terminals
- Open all the terminals to this directory (`learn2soar`)
- In terminal 1 run `./simulate.sh` to start ROS and the gazebo simulation with 
  a plane in it
- In terminal 2 run `./contr_test.sh` to start the attitude controller. It will 
  print a warning whenever it runs at less than 90% of the target frequency 
  (averaged over 10 iterations).
- In terminal 3 run `./learn_to_soar.sh`
- (optional) In terminal 4 run `tensorboard --logdir ./tb_l2s_24/` (replace `tb_l2s_24` 
  with whatever you set as `tensorboard_filename` in `learn_to_soar.py`)
- To interrupt training you only need to `Ctrl-C` in terminal 3.
- If you want to reset the whole simulation you have to interrupt the attitude 
  controller and the training. Then run `./kill.sh` to kill ROS and Gazebo 
  (`Ctrl-c` can leave some processes behind).

# useful cmds

roslaunch rotors_gazebo fixed_wing_hil.launch verbose:=true uav_name:=glider

rostopic pub -r 100 /uav_1/prop_ref_0 std_msgs/Float32 -- 100000

rostopic pub -1 /uav_1/elev_pos std_msgs/Float32 -- 100

rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState "{model_name: 'uav_1', pose: {position: {x: 10, y: 0, z: 55}, orientation: {x: 0, y: 0, z: 0, w: 1}}, twist: {linear: {x: 10, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}, reference_frame: ground_collision_plane}"

rosrun rqt_multiplot rqt_multiplot

rqt_plot /gazebo/model_states/pose[-1]/position/z /glider_roll /glider_roll_ref &