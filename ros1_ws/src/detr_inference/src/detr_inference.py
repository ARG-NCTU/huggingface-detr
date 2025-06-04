#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import os
import rospkg
import time
import cv2
import matplotlib.colors as mcolors

rospack = rospkg.RosPack()

class DetrInferenceNode:
    def __init__(self):
        rospy.init_node('detr_inference', anonymous=True)
        
        self.rospack = rospkg.RosPack()
        self.bridge = CvBridge()
        
        # Load parameters
        self.load_parameters()
        
        # Load model
        self.load_model()
        
        # Load class names and colors
        self.class_list = self.load_classes()
        self.class_colors = {class_name: self.colors[i % len(self.colors)] for i, class_name in enumerate(self.class_list)}
        
        # Publishers
        self.init_publishers()
        
        # Subscriber
        rospy.Subscriber(self.sub_camera_topic, CompressedImage, self.detection_callback)

    def load_parameters(self):
        """Load ROS parameters."""
        self.classes_path = rospy.get_param('~classes_path', os.path.join(self.rospack.get_path("detr_inference"), "classes", "boat_classes.txt"))
        self.hub_id = rospy.get_param('~hub_id', "ARG-NCTU")
        self.repo_id = rospy.get_param('~repo_id', "detr-resnet-50-finetuned-20-epochs-boat-dataset")
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.8)
        self.sub_camera_topic = rospy.get_param('~sub_camera_topic', '/camera_middle/color/image_raw/compressed')
        self.pub_detection_image_enabled = rospy.get_param('~pub_detection_image', True)

        # Colors for bounding boxes
        # self.colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta",
        #                "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint",
        #                "olive", "apricot", "navy", "grey", "white", "black"]
        self.colors = ["yellow",]

    def load_classes(self):
        """Load class names from the specified file."""
        try:
            with open(self.classes_path, "r") as f:
                return [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            rospy.logerr(f"Class file {self.classes_path} not found!")
            return []

    def load_model(self):
        """Load the DETR model and processor."""
        hf_model_path = os.path.abspath(os.path.join(rospack.get_path("detr_inference"), "model", self.hub_id, self.repo_id))
        rospy.loginfo(f"Loading model from {hf_model_path}")
        self.image_processor = AutoImageProcessor.from_pretrained(hf_model_path, local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(hf_model_path, local_files_only=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def init_publishers(self):
        """Initialize ROS publishers."""
        self.pub_detection_image = rospy.Publisher(rospy.get_param('~pub_camera_topic', '/detection_result_img/camera_stitched/compressed'), CompressedImage, queue_size=1)
        self.pub_scores = rospy.Publisher(rospy.get_param('~pub_scores_topic', '~detection_scores'), Float32MultiArray, queue_size=1)
        self.pub_labels = rospy.Publisher(rospy.get_param('~pub_labels_topic', '~detection_labels'), Int32MultiArray, queue_size=1)
        self.pub_boxes = rospy.Publisher(rospy.get_param('~pub_boxes_topic', '~detection_boxes'), Int32MultiArray, queue_size=1)
        self.pub_boxes_bottom = rospy.Publisher(rospy.get_param('~pub_boxes_bottom_topic', '~detection_boxes_bottom'), Int32MultiArray, queue_size=1)
    
    def detect_objects(self, image):
        """Perform object detection on the input image."""
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=self.confidence_threshold, target_sizes=target_sizes)[0]
        return results

    def name_to_bgr(self, color_name):
        """Convert color name string (e.g., 'yellow') to OpenCV BGR tuple."""
        rgb = mcolors.to_rgb(color_name)  # e.g., (1.0, 1.0, 0.0)
        rgb = [int(x * 255) for x in rgb]
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels directly on a cv2 (BGR) image using self.class_colors."""
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            class_name = self.model.config.id2label[label.item()]
            color_name = self.class_colors.get(class_name, "white")
            box_color = self.name_to_bgr(color_name)
            x, y, x2, y2 = [int(i) for i in box.tolist()]
            cv2.rectangle(image, (x, y), (x2, y2), box_color, 2)
            bbox_area = (x2 - x) * (y2 - y)
            text = f"{class_name} {score:.2f} area:{bbox_area}"
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(image, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        return image


    def detection_callback(self, msg):

        start_time = time.time()
        msg_header = msg.header

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.loginfo('CvBridgeError: %s', e)
            return

        rospy.loginfo("Image conversion took: %f seconds", time.time() - start_time)

        pil_image = PILImage.fromarray(cv_image)

        if pil_image:
            start_time = time.time()
            detections = self.detect_objects(pil_image)
            rospy.loginfo("Detection processing took: %f seconds", time.time() - start_time)

            if len(detections["scores"]) > 0:
                scores = detections["scores"].detach().cpu().numpy().tolist()
                labels = detections["labels"].detach().cpu().numpy().tolist()
                boxes = detections["boxes"].detach().cpu().numpy().astype(np.int32).tolist()
                box_bottom = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    center_x = (x_min + x_max) / 2.0
                    bottom_y = y_max
                    box_bottom.extend([center_x, bottom_y])

                # Publish detections
                self.pub_scores.publish(Float32MultiArray(data=scores))
                self.pub_labels.publish(Int32MultiArray(data=labels))
                self.pub_boxes.publish(Int32MultiArray(data=[item for sublist in boxes for item in sublist]))  # Flattened 2D array
                self.pub_boxes_bottom.publish(Int32MultiArray(data=box_bottom))


                rospy.loginfo("Published detections: %d objects detected", len(scores))

            if self.pub_detection_image_enabled:
                try:
                    processed_image = self.draw_detections(cv_image, detections)
                    ros_image = self.bridge.cv2_to_compressed_imgmsg(processed_image, dst_format='jpeg')
                    if msg_header is not None:
                        ros_image.header = msg_header
                        ros_image.header.stamp = rospy.Time.now()
                    self.pub_detection_image.publish(ros_image)
                    rospy.loginfo("Total processing time: %f seconds", time.time() - start_time)
                except CvBridgeError as e:
                    rospy.loginfo('CvBridgeError while converting back: %s', e)

if __name__ == '__main__':
    try:
        node = DetrInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
