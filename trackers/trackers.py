from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment
import sys
import logging

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self, player_model_path, ball_model_path):
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)
        self.tracker = sv.ByteTrack()
        self.player_color_memory = {}
        self.previous_tracks = {}
        logger.info("Tracker initialized")

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames, model, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def update_player_color_memory(self, track_id, color):
        if track_id not in self.player_color_memory:
            self.player_color_memory[track_id] = np.array(color)
        else:
            current_color = self.player_color_memory[track_id]
            if np.linalg.norm(current_color - color) > 10:
                self.player_color_memory[track_id] = 0.9 * self.player_color_memory[track_id] + 0.1 * np.array(color)

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        player_detections = self.detect_frames(frames, self.player_model)
        ball_detections = self.detect_frames(frames, self.ball_model)

        tracks = {
            "Player": [],
            "Ref": [],
            "Shot Clock": [],
            "Time Remaining": [],
            "Team Points": [],
            "Hoop": [],
            "Ball": []
        }

        for frame_num, (player_detection, ball_detection) in enumerate(zip(player_detections, ball_detections)):
            cls_names = player_detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            player_detection_supervision = sv.Detections.from_ultralytics(player_detection)
            player_detection_with_tracks = self.tracker.update_with_detections(player_detection_supervision)

            tracks["Player"].append({})
            tracks["Ref"].append({})
            tracks["Shot Clock"].append({})
            tracks["Time Remaining"].append({})
            tracks["Team Points"].append({})
            tracks["Hoop"].append({})
            tracks["Ball"].append({})

            current_tracks = {}

            for frame_detection in player_detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["Player"]:
                    current_tracks[track_id] = {"bbox": bbox}

                    if "team_color" in current_tracks[track_id]:
                        color = current_tracks[track_id]["team_color"]
                        self.update_player_color_memory(track_id, color)

                if cls_id == cls_names_inv["Ref"]:
                    tracks["Ref"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["Hoop"]:
                    tracks["Hoop"][frame_num][1] = {"bbox": bbox}

                if cls_id == cls_names_inv["Team Points"]:
                    tracks["Team Points"][frame_num][1] = {"bbox": bbox}

                if cls_id == cls_names_inv["Time Remaining"]:
                    tracks["Time Remaining"][frame_num][1] = {"bbox": bbox}

                if cls_id == cls_names_inv["Shot Clock"]:
                    tracks["Shot Clock"][frame_num][1] = {"bbox": bbox}

            for track_id in current_tracks:
                tracks["Player"][frame_num][track_id] = current_tracks[track_id]

            
            self.previous_tracks = self.smooth_tracks(tracks["Player"][frame_num])

            cls_names = ball_detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            ball_detection_supervision = sv.Detections.from_ultralytics(ball_detection)

            highest_confidence_detection = None
            for frame_detection in ball_detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    if highest_confidence_detection is None or confidence > highest_confidence_detection[2]:
                        highest_confidence_detection = frame_detection

            if highest_confidence_detection is not None:
                bbox = highest_confidence_detection[0].tolist()
                tracks["Ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def smooth_tracks(self, current_tracks):
        smoothed_tracks = {}
        for track_id, track_data in current_tracks.items():
            if track_id in self.previous_tracks:
                previous_bbox = np.array(self.previous_tracks[track_id]['bbox'])
                current_bbox = np.array(track_data['bbox'])
                smoothed_bbox = 0.8 * previous_bbox + 0.2 * current_bbox
                smoothed_tracks[track_id] = {'bbox': smoothed_bbox.tolist()}
            else:
                smoothed_tracks[track_id] = track_data
        return smoothed_tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # Bottom of bbox
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            (x_center, y2),
            (int(0.6 * width), int(0.15 * width)),
            0.0,  
            0,    
            360, 
            color,
            2, 
            cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_rectangle(self, frame, bbox, color):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["Player"][frame_num]
            referee_dict = tracks["Ref"][frame_num]
            ball_dict = tracks["Ball"][frame_num]
            hoop_dict = tracks["Hoop"][frame_num]
            team_points_dict = tracks["Team Points"][frame_num]

         
            for track_id, player in player_dict.items():
                if track_id in self.player_color_memory:
                    color = self.player_color_memory[track_id]
                else:
                    color = (0, 0, 255)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            for track_id, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, ref["bbox"], (0, 0, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_rectangle(frame, ball["bbox"], (0, 255, 0))

            for track_id, hoop in hoop_dict.items():
                frame = self.draw_rectangle(frame, hoop["bbox"], (255, 0, 0))

            output_video_frames.append(frame)

        return output_video_frames
