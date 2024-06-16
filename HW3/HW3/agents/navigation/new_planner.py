import numpy as np 
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc
import json
import networkx as nx
import re
import carla
from scipy import spatial

# world to pixel 
def world_2_pixel(min_pos, pos):

    # Pixel coordinates on full map
    x = int(10 * (pos[0] - min_pos[0]))
    y = int(10 * (pos[1] - min_pos[1]))
    
    return [x, y]
    
# pixel to world 
def pixel_2_world(min_pos, pos):
    
    # on carla coordinate     
    x = min_pos[0]  + pos[0]/10.0
    y = min_pos[1]  + pos[1]/10.0
    
    return [x, y]


def read_adjlist_with_tuples(file_path):
    
    G = nx.DiGraph() 
    with open(file_path, 'r') as file:
        for line in file:
            entries = re.findall(r'\([^)]+\)', line)
            if len(entries) == 0:
                continue
            start = entries[0]
            start = start.strip('() ')
            x0, y0 =  map(int, start.split(','))
            start_tuple = tuple([x0, y0])
            
            for entry in entries[1:]:
                end = entry
                end = end.strip('() ')
                x, y =  map(int, end.split(','))
                second_tuple = tuple([x, y])
                G.add_edge(start_tuple, second_tuple)

    return G
  
class proxy_waypoint(object):
  def __init__(self, coordinate):
     self.transform = carla.Transform(carla.Location(x=coordinate[0], y=coordinate[1], z=0), carla.Rotation())

class NewPlanner(object):
  def __init__(self, town):
      self.town = town
      with open("./bird_eye_view/maps/ori_MapBoundaries.json", 'r') as f:
        data = json.load(f)
        
      self.min_x= data[town]["min_x"]
      self.min_y= data[town]["min_y"]
      self.min_pos = [self.min_x, self.min_y]
      
  def trace_route(self, origin, destination):
    ego_pos = [origin.x, origin.y]
    target_pos = [destination.x, destination.y]
    ego_pos_pixel = world_2_pixel(self.min_pos, ego_pos) # pixel coordinate 
    target_loc_pixel = world_2_pixel(self.min_pos, target_pos) # pixel coordinate 
    
    # get graph 
    graph_path = f"./graph/{self.town}.adjlist"
    G = read_adjlist_with_tuples(graph_path) 
    nodes = np.asarray(G.nodes)
    
    # find closest node using KD tree
    start = tuple(nodes[spatial.KDTree(nodes).query([ego_pos_pixel[0], ego_pos_pixel[1]])[1]])
    end = tuple(nodes[spatial.KDTree(nodes).query([target_loc_pixel[0], target_loc_pixel[1]])[1]])
    path = np.asarray( nx.astar_path(G, start,  end)) # in pixel coordinate
    
    
    # # find closest node using KD tree 
    # start = tuple(nodes[spatial.KDTree(nodes).query([ego_pos_pixel[1], ego_pos_pixel[0]])[1]])
    # end = tuple(nodes[spatial.KDTree(nodes).query([target_loc_pixel[1], target_loc_pixel[0]])[1]])
    
    # path = np.asarray( nx.shortest_path(G, start,  end)) # in pixel coordinate 
    # path[:, [1, 0]] = path[:, [0, 1]]
    
    # to world coordinate     
    path = path/10.0 + self.min_pos
    
    route_list = []
    for i in range(len(path)):
        loc = Loc(path[i][0], path[i][1])
        wp = proxy_waypoint(loc)
        # put a None type command into route
        route_list.append((wp, None))
        
    return route_list