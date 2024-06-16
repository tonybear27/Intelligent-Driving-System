import numpy as np 
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc
import cv2 
import json
import networkx as nx
import re
from scipy import spatial
import math 

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

if __name__ == "__main__":

    path = "./dataset/obstacle_Town03_20_2/states/0010.npz"
    town = "Town03"
    instance_image = cv2.imread(f"./instance_goal_maps/{town}.png")
    
    with open(f"./goals/{town}.json", 'r') as f:
        goal_dict = json.load(f)    

    with open("./bird_eye_view/maps/ori_MapBoundaries.json", 'r') as f:
        data = json.load(f)

    min_x= data[town]["min_x"]
    min_y= data[town]["min_y"]
    min_pos = [ min_x, min_y]


    data = np.load(path, allow_pickle=True)['arr_0'].item()
    ego_id = data['ego_id']
    ego_pos = data[ego_id]['location']
    ego_pos = [ego_pos['x'], ego_pos['y']] # world coordinate 
    ego_yaw = data[ego_id]['rotation']['yaw']
    ego_pos_pixel = world_2_pixel(min_pos, ego_pos) # pixel coordinate 
    
    
    ##################
    
    cord_1 = (data[ego_id]['cord_bounding_box']['cord_6'])
    cord_2 = (data[ego_id]['cord_bounding_box']['cord_4'])
    min_point = [(cord_1[0]+cord_2[0])/2.0, (cord_1[1]+cord_2[1])/2.0]
    head_of_ego = np.array([min_point[0], min_point[1]])
    
    rgb = instance_image[ego_pos_pixel[1]][ego_pos_pixel[0]]
    
    b = rgb[0]
    g = rgb[1]
    r = rgb[2]
    
    target_dict = goal_dict[f"{b}_{g}_{r}"]
    
    num_of_goals = target_dict['num_of_goals']
    
    # # for index in range(num_of_goals):
    # #     # print(index)
    # #     print( target_dict[f'{index}'] )
    
    
    
    
    index = 1
    target_loc = target_dict[f'{index}']['loc']
    target_action = target_dict[f'{index}']['action']
    print(target_action)
    
    # ego_pos = np.asarray(ego_pos)
    # target_loc = np.asarray(target_loc)
    
    ego_pos_pixel = world_2_pixel(min_pos, ego_pos) # pixel coordinate 
    target_loc_pixel = world_2_pixel(min_pos, target_loc) # pixel coordinate 
    
    # get graph 
    graph_path = "./graph/Town03.adjlist"
    G = read_adjlist_with_tuples(graph_path) 
    nodes = np.asarray(G.nodes)
    
    # find closest node using KD tree 
    start = tuple(nodes[spatial.KDTree(nodes).query([ego_pos_pixel[1], ego_pos_pixel[0]])[1]])
    end = tuple(nodes[spatial.KDTree(nodes).query([target_loc_pixel[1], target_loc_pixel[0]])[1]])
    
    # print(ego_pos_pixel)
    # print(start)
    # print(" -- ")
    # print(target_loc_pixel)
    # print(end)
    
    path = np.asarray( nx.shortest_path(G, start,  end)) # in pixel coordinate 
    path[:, [1, 0]] = path[:, [0, 1]]
    
    # to world coordinate     
    path = path/10.0 + min_pos
    
    
    route_list = []
    for i in range(len(path)):
        loc = Loc(path[i][0], path[i][1])
        route_list.append(loc)
        
        
    # For vis
    birdview_producer = BirdViewProducer( 
                                        town, 
                                        PixelDimensions(width=300, height=300), 
                                        pixels_per_meter=5)
    
    pos_0 = data[ego_id]["cord_bounding_box"]["cord_0"]
    pos_1 = data[ego_id]["cord_bounding_box"]["cord_4"]
    pos_2 = data[ego_id]["cord_bounding_box"]["cord_6"]
    pos_3 = data[ego_id]["cord_bounding_box"]["cord_2"]

    agent_bbox_list = []
    agent_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                            Loc(x=pos_1[0], y=pos_1[1]), 
                            Loc(x=pos_2[0], y=pos_2[1]), 
                            Loc(x=pos_3[0], y=pos_3[1]), 
                            ])
    
    
    ego_pos = data[ego_id]['location']
    ego_pos = Loc(x=ego_pos['x'], y= ego_pos['y'])
    ego_yaw = data[ego_id]['rotation']['yaw']
    
    birdview: BirdView = birdview_producer.produce( ego_pos, ego_yaw,
                                                    agent_bbox_list, 
                                                    [], # vehicle_bbox_list,
                                                    [], # ped 
                                                    [], # R
                                                    [], # G
                                                    [], # B
                                                    [], # obstacle_bbox_list,
                                                    route_list)

    image = birdview_producer.as_rgb(birdview)

    cv2.imwrite("./tmp.png", image)    
        