import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssignment:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def preprocess_image(self, image):
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image
    
    def get_clustering_model(self, image, k=5):
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        return kmeans

    def adjust_color(self, color, alpha=1.2, beta=30):
        adjusted_color = cv2.convertScaleAbs(np.array([color]), alpha=alpha, beta=beta)
        return adjusted_color[0]

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image = self.preprocess_image(image)

        height, width, _ = image.shape

        x1, y1 = int(width * 0.3), int(height * 0.2)
        x2, y2 = int(width * 0.7), int(height * 0.5)
        roi = image[y1:y2, x1:x2]

        kmeans = self.get_clustering_model(roi, k=5)
        dominant_color = np.mean(kmeans.cluster_centers_, axis=0)

        adjusted_color = self.adjust_color(dominant_color)

        return adjusted_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++")
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        player_color = player_color.reshape(1, -1) 
        team_id = self.kmeans.predict(player_color)[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id
