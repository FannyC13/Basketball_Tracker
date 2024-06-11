from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssignment

def main():
    video_frames = read_video('input_videos/basketball_lakers_pelicans_2.mp4')

    tracker = Tracker("models/best.pt", "models/best_ball_roboflow.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stubs_byteTrack_lakers_pelicans_model_2.pkl")

    tracks["Ball"] = tracker.interpolate_ball_position(tracks["Ball"])

    team_assignment = TeamAssignment()
    team_assignment.assign_team_color(video_frames[0], tracks['Player'][0])

    output_video_frames = []
    for frame_num, player_track in enumerate(tracks['Player']):
        for player_id, track in player_track.items():
            team = team_assignment.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['Player'][frame_num][player_id]['team'] = team
            tracks['Player'][frame_num][player_id]['team_color'] = team_assignment.team_colors[team]
            tracker.update_player_color_memory(player_id, team_assignment.team_colors[team])

        output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_video_byteTrack_lakers_pelicans_model_2.avi')

if __name__ == "__main__":
    main()
