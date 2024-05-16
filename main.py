from utils import read_video, save_video, measure_distance
from trackers import Tracker
import cv2
from team_assig import Teamassigner
from player_ball_assig import PlayerBallAssig
import numpy as np

def main():
    #Read Video
    video_frames = read_video('input_videos/08fd33_3.mp4')

    #Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = Teamassigner()
    team_assigner.assign_team_colors(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    player_ass = PlayerBallAssig()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_box = tracks['ball'][frame_num][1]['bbox']
        ass_player = player_ass.assign_ball_to_player(player_track, ball_box)
        if ass_player != -1:
            tracks['players'][frame_num][ass_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][ass_player]['team'])
        elif team_ball_control:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()
