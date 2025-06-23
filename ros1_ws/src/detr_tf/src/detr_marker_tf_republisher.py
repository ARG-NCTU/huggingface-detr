#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped

class DetrMarkerTFRepublisher:
    def __init__(self):
        rospy.init_node('detr_marker_tf_republisher')

        self.source_frame = rospy.get_param('~source_frame_id', 'base_link')
        self.target_frame = rospy.get_param('~target_frame_id', 'map')

        self.sub_topic = rospy.get_param('~source_marker_array', '/detr/base_link/marker_array')
        self.pub_topic = rospy.get_param('~target_marker_array', '/detr/map/marker_array')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.subscriber = rospy.Subscriber(self.sub_topic, MarkerArray, self.marker_callback, queue_size=10)
        self.publisher = rospy.Publisher(self.pub_topic, MarkerArray, queue_size=10)

    def marker_callback(self, msg):
        try:
            transformed_array = MarkerArray()
            now = rospy.Time.now()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )

            for marker in msg.markers:
                pose_stamped = PoseStamped()
                pose_stamped.header = marker.header
                pose_stamped.pose = marker.pose
                pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

                marker.header.frame_id = self.target_frame
                marker.header.stamp = now
                marker.pose = pose_transformed.pose

                transformed_array.markers.append(marker)

            self.publisher.publish(transformed_array)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF transform failed: {str(e)}")

if __name__ == '__main__':
    try:
        node = DetrMarkerTFRepublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
