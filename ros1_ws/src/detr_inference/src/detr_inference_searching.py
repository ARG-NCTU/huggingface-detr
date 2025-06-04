#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Bool, Int32MultiArray, Float32
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

class DetrInferenceSearchingNode:
    def __init__(self):
        rospy.init_node('detr_inference_searching', anonymous=True)
        
        self.rospack = rospkg.RosPack()
        self.bridge = CvBridge()
        
        # Load parameters
        self.load_parameters()
        self.detected = False
        
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
        hf_model_path = os.path.join(rospack.get_path("detr_inference"), "model", self.hub_id, self.repo_id)
        self.image_processor = AutoImageProcessor.from_pretrained(hf_model_path, local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(hf_model_path, local_files_only=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def init_publishers(self):
        """Initialize ROS publishers."""
        self.pub_detection_image = rospy.Publisher(rospy.get_param('~pub_camera_topic', '/detection_result_img/camera_stitched/compressed'), CompressedImage, queue_size=1)
        self.pub_highest_conf_bbox_center_cord = rospy.Publisher(rospy.get_param('~pub_highest_conf_bbox_center_cord_topic', '~highest_conf_detection_bbox_center_cord'), Float32MultiArray, queue_size=1)
        self.pub_highest_conf_bbox_area = rospy.Publisher(rospy.get_param('~pub_highest_conf_bbox_area_topic', '~highest_conf_detection_bbox_area'), Float32, queue_size=1)
        self.pub_detected = rospy.Publisher(rospy.get_param('~pub_detected_topic', '~detected'), Bool, queue_size=1)
        self.pub_highest_conf_bbox_bottom_xy = rospy.Publisher(rospy.get_param('~pub_highest_conf_bbox_bottom_xy_topic', '~highest_conf_detection_bbox_bottom_xy'), Int32MultiArray, queue_size=1)

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
    
    def draw_detection(self, image, detection, highest_conf_bbox_center_cord, bbox_area):
        score, class_name, box = detection
        color_name = self.class_colors.get(class_name, "white")
        box_color = self.name_to_bgr(color_name)
        x, y, x2, y2 = [int(i) for i in box]
        cv2.rectangle(image, (x, y), (x2, y2), box_color, 2)

        bbox_area = round(bbox_area, 5)
        highest_conf_bbox_center_cord = [round(cord, 3) for cord in highest_conf_bbox_center_cord]

        text = f"class: {class_name}, conf: {score:.2f}, area: {bbox_area}, cord: {highest_conf_bbox_center_cord}"
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
                # Find the score, label, bbox of highest confidence (score) detection
                highest_confidence_idx = torch.argmax(detections["scores"])
                highest_confidence_score = detections["scores"][highest_confidence_idx].item()
                highest_confidence_label = self.model.config.id2label[detections["labels"][highest_confidence_idx].item()]
                highest_confidence_bbox = detections["boxes"][highest_confidence_idx].detach().cpu().numpy().flatten().tolist()
                
                # rospy.loginfo("Highest confidence detection: %s, score: %f, bbox: %s", highest_confidence_label, highest_confidence_score, highest_confidence_bbox)

                if highest_confidence_score > self.confidence_threshold:
                    self.detected = True
                    self.pub_detected.publish(True)
                    # Normalize the bbox center_x, center_y, width, height to [-1, 1]
                    x1, y1, x2, y2 = highest_confidence_bbox

                    bbox_center_x, bbox_center_y, bbox_width, bbox_height = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

                    image_width = pil_image.width
                    image_height = pil_image.height

                    bbox_center_x = (bbox_center_x - image_width / 2) / (image_width / 2)
                    bbox_center_y = -1 * (bbox_center_y - image_height / 2) / (image_height / 2)
                    bbox_area = (bbox_width * bbox_height) / (image_width * image_height)

                    # rospy.loginfo("Normalized bbox: center_x: %f, center_y: %f, width: %f, height: %f", bbox_center_x, bbox_center_y, bbox_width, bbox_height)

                    # Publish the bbox
                    highest_conf_bbox_center_cord = [bbox_center_x, bbox_center_y]
                    self.pub_highest_conf_bbox_center_cord.publish(Float32MultiArray(data=[bbox_center_x, bbox_center_y]))
                    self.pub_highest_conf_bbox_area.publish(Float32(data=bbox_area))

                    self.pub_highest_conf_bbox_bottom_xy.publish(Int32MultiArray(data=[int((x1 + x2) // 2), int(y2)]))

            if self.pub_detection_image_enabled:
                try:
                    if self.detected:
                        detection = [highest_confidence_score, highest_confidence_label, highest_confidence_bbox]
                        processed_image = self.draw_detection(cv_image, detection, highest_conf_bbox_center_cord, bbox_area)
                    else:
                        processed_image = cv_image
                    self.detected = False
                    
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
        node = DetrInferenceSearchingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
