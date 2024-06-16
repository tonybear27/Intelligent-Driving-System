import sys


import carla
import numpy as np
import time
import csv
import os
import logging
import math
import argparse
from numpy import random
from carla import VehicleLightState as vls
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
import json

def spawn_actor_nearby(world, client, distance=100, v_ratio=0.8, pedestrian=10): 


    # get world and spawn points
    # world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()

    waypoint_list = []
    
    for waypoint in spawn_points:
        point = map.get_waypoint(waypoint.location)
        flag = True
        
        waypoint_list.append(waypoint)
            
    seed_4 = int(time.time()) 
    random.seed(seed_4)
    random.shuffle(waypoint_list)
    
    # print(len(waypoint_list))

    # --------------
    # Spawn vehicles
    # --------------
    num_of_vehicles = 0
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    traffic_manager = client.get_trafficmanager()
    # keep distance
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.set_synchronous_mode(True)
    synchronous_master = False
    vehicles_list = []
    walkers_list = []
    all_id = []

    batch = []

    # vehicle = min(math.ceil(len(waypoint_list) * 0.8), vehicles) if len(waypoint_list) >= 20 else len(waypoint_list)
    vehicle = math.ceil(len(waypoint_list) * v_ratio)
    for n, transform in enumerate(waypoint_list):
        if n >= vehicle:
            break
        num_of_vehicles += 1
        
        seed_5 = int(time.time()) 
        seed_5 += 5*num_of_vehicles



        random.seed(seed_5)
        # while(1):
        blueprint = random.choice(blueprints)
            # print('aaaaaa')
            # print(blueprint.get_attribute('number_of_wheels').type)
            # if int(blueprint.get_attribute('number_of_wheels')) == 4:
            #     break
        #print(blueprint)
        
        if blueprint.has_attribute('color'):

            seed_6 = int(time.time())
            
            seed_6 += 6*num_of_vehicles
            random.seed(seed_6)
            
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            
            seed_7 = int(time.time())
            
            seed_7 += 7*num_of_vehicles
            random.seed(seed_7)
            
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        light_state = vls.Position | vls.LowBeam | vls.LowBeam 

        #client.get_world().spawn_actor(blueprint, transform)
        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    all_vehicle_actors = world.get_actors(vehicles_list)
    for v in all_vehicle_actors:
        traffic_manager.ignore_lights_percentage(v, 30)
        traffic_manager.auto_lane_change(v, True)

    

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 0.5      # how many pedestrians will run
    percentagePedestriansCrossing = 0.5     # how many pedestrians will walk through the road
    spawn_points = []

    for i in range(pedestrian):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    batch = []
    walker_speed = []
    n=0
    for spawn_point in spawn_points:
        
        n+=1
        seed_8 = int(time.time())
        seed_8 += 8*n
        random.seed(seed_8)
        
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            seed_9 = int(time.time())

            seed_9 += 9*n
            random.seed(seed_9)
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    # if not True or not synchronous_master:
    #     world.wait_for_tick()
    # else:
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point

        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        #all_actors[i].go_to_location(center)

        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers.' % (len(vehicles_list), len(walkers_list)))
    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(-20.0)
    return vehicles_list, all_actors, all_id
