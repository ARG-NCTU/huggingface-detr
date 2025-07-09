#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Bool, Int32MultiArray, Float32
from geometry_msgs.msg import PolygonStamped
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
import ast
import math
from visualization_msgs.msg import Marker, MarkerArray
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from huggingface_hub import login

rospack = rospkg.RosPack()

class DetrInferenceSearchingDistanceNode:
    def __init__(self):
        rospy.init_node('detr_inference_distance', anonymous=True)

        login(token=os.environ["HUGGINGFACE_TOKEN"])
        
        self.rospack = rospkg.RosPack()
        self.bridge = CvBridge()
        
        # Load parameters
        self.load_parameters()
        self.detected = False
        
        # Load model
        self.load_model()
        
        # Load class names and colors
        self.class_list = self.load_classes()
        # self.class_colors = {class_name: self.colors[i % len(self.colors)] for i, class_name in enumerate(self.class_list)}
        
        # Publishers
        self.init_publishers()
        
        # Subscriber
        rospy.Subscriber(self.sub_camera_topic, CompressedImage, self.detection_callback)
        rospy.Subscriber(self.sub_camera_annotated_topic, CompressedImage, self.annotated_image_callback)

        self.horizon_polygons_by_camera = {'left': [], 'mid': [], 'right': []}
        rospy.Subscriber('/horizon_points_poly', PolygonStamped, self.horizon_poly_callback)

        self.lastest_annotated_image = None

        self.cam_ext_rotation = [-60, 0, 60]
        self.hfov = 60.0

        self.fixed_id_colors = {
            0: self.name_to_bgr("red"),
            1: self.name_to_bgr("orange"),
            2: self.name_to_bgr("yellow"),
            3: self.name_to_bgr("green"),
        }

        self.previous_objects = {}
        self.tracked_ids_in_this_frame = set()
        self.id_counter = 0
        
        self.detection_frame = 1

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_detection_time = time.time()
        self.marker_delete_timer = rospy.Timer(rospy.Duration(0.5), self.check_detection_timeout)
        

    def get_fixed_color_for_id(self, obj_id):
        return self.fixed_id_colors[obj_id % 4]

    def load_parameters(self):
        """Load ROS parameters."""
        self.classes_path = rospy.get_param('~classes_path', os.path.join(self.rospack.get_path("detr_inference"), "classes", "KS_Buoy_classes.txt"))
        self.hub_id = rospy.get_param('~hub_id', "ARG-NCTU")
        self.repo_id = rospy.get_param('~repo_id', "detr-resnet-50-finetuned-600-epochs-KS-Buoy-dataset")
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.8)
        self.sub_camera_topic = rospy.get_param('~sub_camera_topic', '/camera_pano_stitched/color/image_raw/compressed')
        self.sub_camera_annotated_topic = rospy.get_param('~sub_camera_annotated_topic', '/camera_pano_masked/image_raw/compressed')
        self.pub_detection_image_enabled = rospy.get_param('~pub_detection_image', True)
        self.M_L = np.load(rospy.get_param("~h1_path"), None)
        self.M_R = np.load(rospy.get_param("~h2_path"), None)
        crop_str = rospy.get_param("~crop_rect", None)
        # TF
        self.source_frame = rospy.get_param('~source_frame_id', 'base_link')
        self.target_frame = rospy.get_param('~target_frame_id', 'map')
        self.markers_scale = float(rospy.get_param('~markers_scale', 100))
        # Avoid delaying
        self.keep_frame_ratio = min(rospy.get_param('~keep_frame_ratio', 0.3), 1.0)
        self.keep_frame = int(1 / self.keep_frame_ratio)
        # ID tracking
        self.max_missed_sec = rospy.get_param('~max_missed_sec', 0.5)
        self.max_distance_threshold = rospy.get_param('~max_distance_threshold', 50)
        self.max_id = rospy.get_param('~max_id', 10)

        self.timeout_no_detection = rospy.get_param('~timeout_no_detection', 2.0)
        
        self.image_width = None
        if crop_str:
            try:
                self.crop_rect = ast.literal_eval(crop_str)
                self.image_width = self.crop_rect[2]
                if not isinstance(self.crop_rect, list) or len(self.crop_rect) != 4:
                    rospy.logwarn("Invalid crop_rect format. Expected [x,y,w,h].")
                    self.crop_rect = None
            except Exception as e:
                rospy.logwarn(f"Failed to parse crop_rect: {e}")
                self.crop_rect = None
        else:
            self.crop_rect = None


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
        self.pub_detection_image = rospy.Publisher(rospy.get_param('~pub_camera_topic', '/detr/compressed'), CompressedImage, queue_size=1)
        self.pub_marker_array = rospy.Publisher(rospy.get_param('~pub_marker_array_topic', '/detr/marker_array'), MarkerArray, queue_size=1)
    
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
    
    def draw_detections(self, image, matched_results):
        for obj_id, score, class_name, box, distance, angle in matched_results:
            box_color = self.get_fixed_color_for_id(obj_id)
            x, y, x2, y2 = [int(i) for i in box]
            cv2.rectangle(image, (x, y), (x2, y2), box_color, 2)

            # ID text
            text_id = f"id: {obj_id}"
            text_id_y = y - 60
            cv2.putText(image, text_id, (x, text_id_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, text_id, (x, text_id_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Class text
            text_top_class = f"class: {class_name}"
            text_top_class_y = y - 40
            cv2.putText(image, text_top_class, (x, text_top_class_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, text_top_class, (x, text_top_class_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Confidence text
            text_top_conf = f"conf: {score:.2f}"
            text_top_conf_y = text_top_class_y + 20
            cv2.putText(image, text_top_conf, (x, text_top_conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, text_top_conf, (x, text_top_conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Distance text
            distance = int(distance) if distance is not None else "N/A"
            text_buttom_distance = f"dist: {distance} m"
            text_buttom_distance_y = y2 + 20 if y2 + 20 < image.shape[0] - 10 else y - 20
            cv2.putText(image, text_buttom_distance, (x, text_buttom_distance_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, text_buttom_distance, (x, text_buttom_distance_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Angle text
            angle = int(angle) if angle is not None else "N/A"
            text_buttom_angle = f"angle: {angle} deg"
            text_buttom_angle_y = text_buttom_distance_y + 20 if text_buttom_distance_y + 20 < image.shape[0] - 10 else y - 40
            cv2.putText(image, text_buttom_angle, (x, text_buttom_angle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, text_buttom_angle, (x, text_buttom_angle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        return image

    def horizon_poly_callback(self, msg):
        cam_names = ['left', 'mid', 'right']
        self.horizon_polygons_by_camera = {cam: [] for cam in cam_names}
        points = msg.polygon.points

        num_per_cam = 4
        for i, cam in enumerate(cam_names):
            start = i * num_per_cam
            end = start + num_per_cam
            for p in points[start:end]:
                self.horizon_polygons_by_camera[cam].append((p.x, p.y, p.z))

    def annotated_image_callback(self, msg):
        self.lastest_annotated_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
    def detection_callback(self, msg):
        if self.detection_frame >= self.keep_frame:
            self.detection_frame = 1
        else:
            self.detection_frame += 1
            return
        
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

            if len(detections["scores"]) > 0:
                scores = detections["scores"].detach().cpu().numpy().tolist()
                labels = detections["labels"].detach().cpu().numpy().tolist()
                boxes = detections["boxes"].detach().cpu().numpy().astype(np.int32).tolist()
                
                # rospy.loginfo(f"scores: {scores}, labels: {labels}, boxes: {boxes}")
                
                self.current_bboxes = []
                for score, label_id, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if cx < self.image_width / 3:
                        cam_idx = 0
                    elif cx < 2 * self.image_width / 3:
                        cam_idx = 1
                    else:
                        cam_idx = 2

                    mapped_cord = self.pano_to_original((cx, y2), cam_idx, self.image_width)
                    if mapped_cord is None:
                        continue

                    original_x, original_y = mapped_cord
                    distance = self.get_distance(original_y, cam_idx)
                    angle = self.get_angle(self.image_width, original_x, cam_idx)

                    if distance is None or angle is None:
                        continue

                    class_name = self.model.config.id2label[label_id]
                    self.current_bboxes.append((cx, cy, (score, class_name, box, distance, angle)))
                
                # If no object in previous frame
                if not self.previous_objects:
                    rospy.loginfo("First frame - assigning new IDs to all objects")
                    matched_results = []
                    for cx, cy, info in self.current_bboxes:
                        new_id = self.get_next_available_id()
                        score, class_name, box, distance, angle = info
                        self.previous_objects[new_id] = {
                            'kf': self.create_kalman_filter(cx, cy),
                            'last_seen_time': time.time(),
                            'box': box
                        }
                        matched_results.append((new_id, score, class_name, box, distance, angle))

                    self.tracked_ids_in_this_frame = set([r[0] for r in matched_results])
                    return
                
                # Perform matching
                matched_results = self.match_bboxes_by_distance_and_iou()
                
                if matched_results:
                    self.detected = True
                    marker_array = MarkerArray()

                    # Clear previous markers
                    delete_marker = Marker()
                    delete_marker.action = Marker.DELETEALL
                    marker_array.markers.append(delete_marker)

                    for detection in matched_results:
                        obj_id, score, class_name, box, distance, angle = detection

                        pose_stamped = PoseStamped()
                        pose_stamped.header.frame_id = self.target_frame
                        pose_stamped.header.stamp = rospy.Time.now()
                        pose_stamped.pose.position.x = float(distance)
                        pose_stamped.pose.position.y = -float(distance * math.tan(math.radians(angle)))
                        pose_stamped.pose.position.z = 0.0
                        pose_stamped.pose.orientation.w = 1.0

                        try:
                            transform = self.tf_buffer.lookup_transform(
                                self.target_frame,
                                self.source_frame,
                                rospy.Time(0),
                                rospy.Duration(0.5)
                            )
                            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
                        except Exception as e:
                            rospy.logwarn(f"Transform failed for obj_id {obj_id}: {e}")
                            return

                        marker = Marker()
                        marker.header.frame_id = self.target_frame
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "detected_object"
                        marker.id = obj_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        marker.pose = pose_transformed.pose

                        marker.scale.x = self.markers_scale
                        marker.scale.y = self.markers_scale
                        marker.scale.z = self.markers_scale
                        bgr = self.get_fixed_color_for_id(obj_id)
                        marker.color.b = bgr[0] / 255.0
                        marker.color.g = bgr[1] / 255.0
                        marker.color.r = bgr[2] / 255.0
                        marker.color.a = 0.7

                        marker_array.markers.append(marker)

                    self.pub_marker_array.publish(marker_array)

                
                rospy.loginfo("Detection processing took: %f seconds", time.time() - start_time)

            if self.pub_detection_image_enabled:
                try:
                    if self.detected:
                        processed_image = self.draw_detections(self.lastest_annotated_image, matched_results)
                    else:
                        processed_image = self.lastest_annotated_image
                    self.detected = False
                    
                    ros_image = self.bridge.cv2_to_compressed_imgmsg(processed_image, dst_format='jpeg')
                    if msg_header is not None:
                        ros_image.header = msg_header
                        ros_image.header.stamp = rospy.Time.now()
                    self.pub_detection_image.publish(ros_image)
                    rospy.loginfo("Total processing time: %f seconds", time.time() - start_time)
                except CvBridgeError as e:
                    rospy.loginfo('CvBridgeError while converting back: %s', e)
    
    def check_detection_timeout(self, event):
        if time.time() - self.last_detection_time > self.timeout_no_detection:
            rospy.logwarn("Too long without detection — sending DELETEALL marker.")
            marker_array = MarkerArray()
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.pub_marker_array.publish(marker_array)
    
    def unwarp_point(self, pano_pt, M):
        if M is None or pano_pt is None or np.any(np.isnan(pano_pt)):
            return None
        x, y = pano_pt
        M33 = np.vstack([M, [0, 0, 1]]) if M.shape == (2, 3) else M
        try:
            M_inv = np.linalg.inv(M33)
        except np.linalg.LinAlgError:
            return None
        p = np.array([x, y, 1.0], dtype=np.float32)
        src_p = M_inv @ p
        if abs(src_p[2]) < 1e-6:
            return None
        return (src_p[0] / src_p[2], src_p[1] / src_p[2])

    def pano_to_original(self, pt_shifted, cam_idx, w):
        crop_x, crop_y = 0, 0
        if self.crop_rect:
            crop_x, crop_y = self.crop_rect[:2]
        pano_x, pano_y = pt_shifted[0] + crop_x, pt_shifted[1] + crop_y
        pano_x -= w
        if cam_idx == 0:
            return self.unwarp_point((pano_x, pano_y), self.M_L)
        elif cam_idx == 1:
            return (pano_x, pano_y)
        elif cam_idx == 2:
            return self.unwarp_point((pano_x, pano_y), self.M_R)
        
    def get_distance(self, y, cam_idx):
        cam_names = ['left', 'mid', 'right']
        cam = cam_names[cam_idx]
        if cam not in self.horizon_polygons_by_camera:
            return None
        points = self.horizon_polygons_by_camera[cam]
        if not points or len(points) < 2:
            return None

        points_sorted = sorted(points, key=lambda p: p[1], reverse=True)

        for i in range(len(points_sorted) - 1):
            y1, y2 = points_sorted[i][1], points_sorted[i + 1][1]
            z1, z2 = points_sorted[i][2], points_sorted[i + 1][2]

            if (y1 >= y >= y2) or (y2 >= y >= y1):
                if abs(y2 - y1) < 1e-6:
                    return (z1 + z2) / 2.0
                ratio = (y - y1) / (y2 - y1)
                z = z1 + ratio * (z2 - z1)
                return z

        return None
    
    def get_angle(self, image_width, x, cam_idx):
        if not (0 <= cam_idx < len(self.cam_ext_rotation)):
            rospy.logwarn(f"Invalid cam_idx: {cam_idx}")
            return None

        cam_center_angle = self.cam_ext_rotation[cam_idx]
        half_fov = self.hfov / 2.0
        angle_min = cam_center_angle - half_fov
        angle_max = cam_center_angle + half_fov

        x = max(0, min(x, image_width))
        x_ratio = x / image_width
        
        angle = angle_min + (angle_max - angle_min) * x_ratio
        return angle

    @staticmethod
    def compute_iou(box1, box2):
        # box: (x1, y1, x2, y2)
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        unionArea = box1Area + box2Area - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def create_kalman_filter(self, cx, cy):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  

        kf.F = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        kf.x = np.array([cx, cy, 0, 0])  
        kf.P *= 1000.
        kf.R *= 10.
        kf.Q = np.eye(4)

        return kf

    def match_bboxes_by_distance_and_iou(self):
        """
        prev_boxes: list of (id, (cx, cy))
        curr_boxes: list of (cx, cy, (score, class_name, box, distance, angle))
        return: list of (id, score, class_name, box, distance, angle)
        """
        matched_result = []
        self.tracked_ids_in_this_frame = set()

        prev_boxes = []
        for obj_id, obj in self.previous_objects.items():
            kf = obj['kf']
            kf.predict()
            pred_cx, pred_cy = kf.x[:2]
            prev_boxes.append((obj_id, (pred_cx, pred_cy)))

        curr_boxes = self.current_bboxes

        num_prev = len(prev_boxes)
        num_curr = len(curr_boxes)
        if num_prev == 0 or num_curr == 0:
            rospy.loginfo(f"Skipping match: num_prev={num_prev}, num_curr={num_curr}")
            return matched_result

        cost_matrix = np.full((num_prev, num_curr), fill_value=np.inf)

        for i, (pid, (pcx, pcy)) in enumerate(prev_boxes):
            prev_box = self.previous_objects[pid]['box']  # You must store 'box' in previous_objects in advance
            for j, (ccx, ccy, info) in enumerate(curr_boxes):
                score, class_name, curr_box, distance, angle = info
                dist = math.hypot(ccx - pcx, ccy - pcy)
                if dist > self.max_distance_threshold:
                    continue  # Skip if distance is too large

                norm_dist = dist / self.max_distance_threshold  # Normalize to 0~1
                iou = self.compute_iou(prev_box, curr_box)
                cost = norm_dist + (1 - iou)
                cost_matrix[i, j] = cost

        if np.all(np.isinf(cost_matrix)):
            rospy.loginfo("All costs are inf — skipping matching for this frame.")
            return matched_result

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # rospy.loginfo(f"row indexes: {row_ind}, col_indexes: {col_ind}")

        used_curr_indices = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] == np.inf:
                continue  # Invalid match

            pid = prev_boxes[i][0]
            cx, cy, info = curr_boxes[j]
            score, class_name, box, distance, angle = info

            matched_result.append((pid, score, class_name, box, distance, angle))

            kf = self.previous_objects[pid]['kf']
            kf.update(np.array([cx, cy]))

            self.previous_objects[pid]['last_seen_time'] = time.time()
            self.previous_objects[pid]['box'] = box

            self.tracked_ids_in_this_frame.add(pid)
            used_curr_indices.add(j)

        # For unmatched current boxes, assign new IDs
        for j, (cx, cy, info) in enumerate(curr_boxes):
            if j in used_curr_indices:
                continue
            new_id = self.get_next_available_id()
            if new_id == -1:
                continue
            score, class_name, box, distance, angle = info
            matched_result.append((new_id, score, class_name, box, distance, angle))
            self.previous_objects[new_id] = {
                'kf': self.create_kalman_filter(cx, cy),
                'last_seen_time': time.time(),
                'box': box
            }
            self.tracked_ids_in_this_frame.add(new_id)

        # Update missed count for unmatched previous objects
        for obj_id in list(self.previous_objects.keys()):
            if obj_id not in self.tracked_ids_in_this_frame:
                if time.time() - self.previous_objects[obj_id]['last_seen_time'] > self.max_missed_sec:
                    del self.previous_objects[obj_id]

        return matched_result
    
    def get_next_available_id(self):
        used_ids = set(self.previous_objects.keys()) | self.tracked_ids_in_this_frame
        for i in range(self.max_id):
            if i not in used_ids:
                return i
        rospy.logwarn("No available ID found. Consider increasing max_id.")
        return -1
        
if __name__ == '__main__':
    try:
        node = DetrInferenceSearchingDistanceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
