#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import math
import time

class FakeTFPublisher:
    def __init__(self):
        rospy.init_node('fake_tf_publisher')
        self.br = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(10)  # 10Hz

        self.start_time = time.time()
        self.square_length = 5.0  # meters
        self.speed = 1.0          # meters/sec
        self.period = 4 * self.square_length / self.speed  # total time to complete square

    def get_position_on_square(self, t):
        # Normalize time to square period
        t = t % self.period
        segment_time = self.square_length / self.speed

        if t < segment_time:
            x = t * self.speed
            y = 0.0
            yaw = 0.0
        elif t < 2 * segment_time:
            x = self.square_length
            y = (t - segment_time) * self.speed
            yaw = math.pi / 2
        elif t < 3 * segment_time:
            x = self.square_length - (t - 2 * segment_time) * self.speed
            y = self.square_length
            yaw = math.pi
        else:
            x = 0.0
            y = self.square_length - (t - 3 * segment_time) * self.speed
            yaw = -math.pi / 2

        return x, y, yaw

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = time.time() - self.start_time

            x, y, yaw = self.get_position_on_square(elapsed)

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = 0.0
            q = self.yaw_to_quaternion(yaw)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.br.sendTransform(t)
            self.rate.sleep()

    def yaw_to_quaternion(self, yaw):
        # Convert a yaw angle (in radians) to quaternion (x, y, z, w)
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

if __name__ == '__main__':
    try:
        node = FakeTFPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
