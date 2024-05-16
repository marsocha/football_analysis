import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssig():

    def __init__(self):
        self.max_player_distance = 30

    def assign_ball_to_player(self, players, ball_bbox):

        ball_center = get_center_of_bbox(ball_bbox)
        minimum_distance = 99999
        ass_player = -1
        for player_id, player in players.items():
            player_bbox = player['bbox']

            y2 = int(player_bbox[3])
            x_center= get_center_of_bbox(player_bbox)[0]
            distance = measure_distance((x_center, y2), ball_center)
            # distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_center)
            # distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_center)
            # distance = min(distance_left,distance_right)
            if distance < self.max_player_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    ass_player = player_id
        return ass_player
