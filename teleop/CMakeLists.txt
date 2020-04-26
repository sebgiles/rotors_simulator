cmake_minimum_required(VERSION 2.8.3)
project(teleop)

find_package(catkin REQUIRED
    COMPONENTS cmake_modules message_generation std_msgs)

add_message_files(
    FILES
    TwoTuple.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
    CATKIN_DEPENDS message_runtime
)

install(PROGRAMS scripts/mouse_teleop.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

foreach(dir config launch)
  install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach()
