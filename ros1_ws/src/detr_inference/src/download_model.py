#!/usr/bin/env python3
import os
import rospy
import rospkg
from huggingface_hub import hf_hub_download
from huggingface_hub import login

class HFModelDownloader:
    def __init__(self):
        rospy.init_node("hf_model_downloader", anonymous=True)

        login(token=os.environ["HUGGINGFACE_TOKEN"])

        self.account_name = rospy.get_param("~hf_account_name", None)
        self.repo_name = rospy.get_param("~hf_repo_name", None)

        if not self.account_name or not self.repo_name:
            rospy.logerr("Missing ROS parameters: ~hf_account_name or ~hf_repo_name.")
            raise ValueError("Both ~hf_account_name and ~hf_repo_name are required.")

        self.model_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
        self.local_model_path = self.get_local_model_path()

    def get_local_model_path(self):
        rospack = rospkg.RosPack()
        base_dir = os.path.join(rospack.get_path("detr_inference"), "model")
        model_path = os.path.join(base_dir, self.account_name, self.repo_name)
        os.makedirs(model_path, exist_ok=True)
        return model_path

    def download_model(self):
        if all(os.path.exists(os.path.join(self.local_model_path, f)) for f in self.model_files):
            rospy.loginfo(f"Model already exists at: {self.local_model_path}")
            return

        rospy.loginfo(f"Downloading model from {self.account_name}/{self.repo_name} to {self.local_model_path}")
        for file in self.model_files:
            hf_hub_download(
                repo_id=f"{self.account_name}/{self.repo_name}",
                repo_type="model",
                filename=file,
                local_dir=self.local_model_path
            )
        rospy.loginfo("Download complete.")

if __name__ == "__main__":
    try:
        downloader = HFModelDownloader()
        downloader.download_model()
    except Exception as e:
        rospy.logerr(f"Model download failed: {e}")
