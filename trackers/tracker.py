from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append('./')
from utils import get_center_of_bbox, get_bbox_width
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker =sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {'bbox' : x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
        return  detections

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center = (x_center, y2), 
                    axes = (int(width), int(0.3 * width)),
                    angle = 0.0,
                    startAngle = -45,
                    endAngle = 225,
                    color = color,
                    thickness = 3,
                    lineType = cv2.LINE_4
                    )
        
        rect_width = 40
        rect_high = 20
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        y1_rect = (y2 - rect_high // 2) + 10
        y2_rect = (y2 + rect_high // 2) + 10 

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 10
            if track_id > 99:
                x_1_text = -10
            cv2.putText(frame, str(track_id), (x1_rect + 5, y2_rect - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    #draw any writing you want in the frame where you want
    def draw_writing(self, frame, bbox, color, writing ):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.putText(frame, writing, (x_center, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)  
        return frame

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }
        for frame_num, detections in enumerate(detections):
            cls_names = detections.names
            cls_names_inv = {v:k for k, v, in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detections)

            # Convert Goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']
            print(detection_supervision)

            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overaly = frame.copy()
        cv2.rectangle(overaly, (0, 0), (frame.shape[1], 50), (0, 0, 0), cv2.FILLED)
        alpha = 0.5
        cv2.addWeighted(overaly, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        cv2.putText(frame, f'Team 1: {team_1*100:.2f}%', (1400, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Team 2: {team_2*100:.2f}%', (1400, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num % 100 == 0:
                print(f'Processing frame {frame_num}/{len(video_frames)}')
            frame = frame.copy()
        
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 0))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

            #Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (120, 0, 255))

            #Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_writing(frame, ball['bbox'], (0, 0, 255), "Ball")

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)   
        
        return output_video_frames
