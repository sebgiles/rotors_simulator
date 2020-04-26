#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Enrique Fernandez
# Released under the BSD License.
#
# Authors:
#   * Enrique Fernandez

import Tkinter

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Bool
import numpy


class MouseTeleop():
    def __init__(self):
        # Retrieve params:
        self._frequency = rospy.get_param('~frequency', 0.0)
        self._scale = rospy.get_param('~scale', 1.0)

        # Create twist publisher:
        self._pub_cmd = rospy.Publisher('teleop_mouse_cmd', TwoTuple, queue_size=1)
        self._pub_pressed = rospy.Publisher('teleop_mouse_pressed', Bool, queue_size=1)

        # Initialize Command components to zero:
        self._h_cmd = 0.0
        self._v_cmd = 0.0

        # Initialize mouse position (x, y) to None (unknown); it's initialized
        # when the mouse button is pressed on the _start callback that handles
        # that event:
        self._x = None
        self._y = None

        # Create window:
        self._root = Tkinter.Tk()
        self._root.title('Mouse Teleop')

        # Make window non-resizable:
        self._root.resizable(0, 0)

        # Create canvas:
        self._canvas = Tkinter.Canvas(self._root, bg='white')

        # Create canvas objects:
        self._canvas.create_arc(0, 0, 0, 0, fill='red', outline='red',
                width=1, style=Tkinter.PIESLICE, start=90.0, tag='h_cmd')
        self._canvas.create_line(0, 0, 0, 0, fill='blue', width=4, tag='v_cmd')

        # Create canvas text objects:
        self._text_v_cmd = Tkinter.StringVar()
        self._text_h_cmd = Tkinter.StringVar()

        self._lab_v_cmd = Tkinter.Label(self._root,
                anchor=Tkinter.W, textvariable=self._text_v_cmd)
        self._lab_h_cmd = Tkinter.Label(self._root,
                anchor=Tkinter.W, textvariable=self._text_h_cmd)

        self._text_v_cmd.set('v = %0.2f ' % self._v_cmd)
        self._text_h_cmd.set('h = %0.2f ' % self._h_cmd)

        self._lab_v_cmd.pack()
        self._lab_h_cmd.pack()

        # Bind event handlers:
        self._canvas.bind('<Button-1>', self._start)
        self._canvas.bind('<ButtonRelease-1>', self._release)

        self._canvas.bind('<Configure>', self._configure)

        self._canvas.bind('<B1-Motion>', self._mouse_motion)

        self._canvas.pack()

        # If frequency is positive, use synchronous publishing mode:
        if self._frequency > 0.0:
            # Create timer for the given frequency to publish the twist:
            period = rospy.Duration(1.0 / self._frequency)

            self._timer = rospy.Timer(period, self._publish)

        # Start window event manager main loop:
        self._root.mainloop()


    def __del__(self):
        if self._frequency > 0.0:
            self._timer.shutdown()
        self._root.quit()


    def _start(self, event):
        self._x, self._y = event.y, event.x
        self._h_cmd = self._v_cmd = 0
        self._pub_pressed.publish(Bool(True))


    def _release(self, event):
        self._h_cmd = self._v_cmd = 0.0
        self._publish()
        self._pub_pressed.publish(Bool(False))
        self._update_gui()


    def _configure(self, event):
        self._width, self._height = event.height, event.width
        self._c_x = self._height / 2.0
        self._c_y = self._width  / 2.0
        self._r = min(self._height, self._width) * 0.25


    def _mouse_motion(self, event):
        dy, dx = self._relative_motion(event.y, event.x)
        self._update_cmd(dx, dy) 
        self._publish()
        self._update_gui()


    def _relative_motion(self, x, y):
        dx = self._x - x
        dy = self._y - y

        dx /= float(self._width)
        dy /= float(self._height)

        dx = max(-1.0, min(dx, 1.0))
        dy = max(-1.0, min(dy, 1.0))

        return dx, dy


    def _update_cmd(self, dx, dy):
        self._h_cmd = dx * self._scale
        self._v_cmd = dy * self._scale


    def _update_coords(self, tag, x0, y0, x1, y1):
        x0 += self._c_x
        y0 += self._c_y

        x1 += self._c_x
        y1 += self._c_y

        self._canvas.coords(tag, (x0, y0, x1, y1))


    def _draw_v(self, v):
        x = -v * float(self._width)
        self._update_coords('v_cmd', 0, 0, 0, x)


    def _draw_h(self, w):
        x0 = y0 = -self._r
        x1 = y1 =  self._r

        self._update_coords('h_cmd', x0, y0, x1, y1)

        yaw = w * numpy.rad2deg(self._scale)

        self._canvas.itemconfig('h_cmd', extent=yaw)


    def _update_gui(self):
        v = self._v_cmd / self._scale
        h = self._h_cmd / self._scale

        self._draw_v(v)
        self._draw_h(h)

        self._text_v_cmd.set('v = %0.2f ' % self._v_cmd)
        self._text_h_cmd.set('h = %0.2f ' % self._h_cmd)
        

    def _publish(self, event=None):
        cmd = TwoTuple()
        cmd.v = self._v_cmd
        cmd.h = self._h_cmd
        self._pub_cmd.publish(cmd)




def main():
    rospy.init_node('mouse_teleop')

    MouseTeleop()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
