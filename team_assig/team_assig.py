from sklearn.cluster import KMeans

class Teamassigner:
        def __init__(self):

                self.team_colors = {}
                self.players_team_dict = {}

        def get_clustering_model(self, image):

                im2d = image.reshape((-1, 3))
                kmeans = KMeans(n_clusters=2, init = "k-means++", random_state=0).fit(im2d)
        
                return kmeans
        
        def get_player_color(self, frame, bbox):

                image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                top_half = image[0:int(image.shape[0]/2), :]
                kmeans = self.get_clustering_model(top_half)
                labels = kmeans.labels_
                cluster_img = labels.reshape(top_half.shape[0], top_half.shape[1])
                corner_clusters = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
                non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
                player_color = 1 - non_player_cluster
                player_color = kmeans.cluster_centers_[player_color]
                
                return player_color

        def assign_team_colors(self, frame, player_detections):

                player_colors = []
                for _, player_detection in player_detections.items():
                        bbox = player_detection['bbox']
                        player_color = self.get_player_color(frame, bbox)
                        player_colors.append(player_color)
                kmeans = KMeans(n_clusters=2, init = "k-means++", n_init = 10).fit(player_colors)
                self.kmeans = kmeans

                self.team_colors[1] = kmeans.cluster_centers_[0]
                self.team_colors[2] = kmeans.cluster_centers_[1]

        def get_player_team(self, frame, player_bbox, player_id):

                if player_id in self.players_team_dict:
                        return self.players_team_dict[player_id]
                player_color = self.get_player_color(frame, player_bbox)
                team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
                team_id += 1
                if player_id == 117:
                        team_id = 2 #gk number
                self.players_team_dict[player_id] = team_id

                return team_id
