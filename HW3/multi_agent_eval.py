#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.


from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
# try:
# 	sys.path.append(glob.glob('/home/hcis-s15/Documents/Homework/IDS_s24/HW0/carla_14/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
# 		sys.version_info.major,
# 		sys.version_info.minor,
# 		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
# 	pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import argparse
import carla
import cv2
import datetime
import importlib
import json
import logging
import math
import pygame
import random
import re
import numpy as np
import time

from autopilot import AutoPilot
from agents.navigation.global_route_planner import GlobalRoutePlanner, get_proxy_route
from agents.navigation.new_planner import NewPlanner
from roach_agent import BEV_MAP
from threading import Thread
from sensors import SensorManager, CollisionSensor, get_actor_display_name
from plant_agent import PlanTAgent
from pathlib import Path
from PIL import Image
from checkpoint_tools import parse_checkpoint
# from SRL_agent import SRLAgent

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
	rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
	name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
	presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
	return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
	def __init__(self, carla_world, hud, args, center):
		self.world = carla_world
		settings = self.world.get_settings()
		settings.fixed_delta_seconds = 0.05
		settings.synchronous_mode = True  # Enables synchronous mode
		self.world.apply_settings(settings)
		self.actor_role_name = args.rolename
		self.rend = int(os.environ.get("REND", 0))		

		try:
			self.map = self.world.get_map()
			self.town = self.map.name.split('/')[-1]
		except RuntimeError as error:
			print('RuntimeError: {}'.format(error))
			print('  The server could not send the OpenDRIVE (.xodr) file:')
			print('  Make sure it exists, has the same name of your town, and is correct.')
			sys.exit(1)
		
		self.world.unload_map_layer(carla.MapLayer.Buildings)     
		self.world.unload_map_layer(carla.MapLayer.Decals)     
		self.world.unload_map_layer(carla.MapLayer.Foliage)     
		self.world.unload_map_layer(carla.MapLayer.Ground)     
		self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)         
		self.world.unload_map_layer(carla.MapLayer.Particles)     
		self.world.unload_map_layer(carla.MapLayer.Props)     
		self.world.unload_map_layer(carla.MapLayer.StreetLights)     
		self.world.unload_map_layer(carla.MapLayer.Walls)     

			
		self.hud = hud
		self.player = None
		self.gnss_sensor = None
		self.imu_sensor = None
		self.camera_manager = None
		self._weather_presets = find_weather_presets()
		self._weather_index = 0
		self._actor_filter = args.filter
  
		self.sensor_spec = [{
				'type': 'sensor.camera.rgb',
        'x': center[0],
        'y': center[1],
        'z': 50,
        'roll': 0,
        'pitch': -90,
        'yaw': center[2],
        'width': 512,
        'height': 512,
        'fov': 60,
        'id': 'rgb_center'
			}]
  
		self.restart()
		self.world.on_tick(hud.on_world_tick)
		self.recording_enabled = False
		self.recording_start = 0
		self.constant_velocity_enabled = False
		self.current_map_layer = 0
		self.surface = None

	def restart(self):
		self.player_max_speed = 1.3 #1.589
		self.player_max_speed_fast = 3.713
		# Keep same camera config if the camera manager exists.
		# cam_index = self.camera_manager.index if self.camera_manager is not None else 0
		# cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
		# Get a random blueprint.
		blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
		blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
		blueprint.set_attribute('role_name', self.actor_role_name)
		if blueprint.has_attribute('color'):
			color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', color)
		if blueprint.has_attribute('driver_id'):
			driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
			blueprint.set_attribute('driver_id', driver_id)
		if blueprint.has_attribute('is_invincible'):
			blueprint.set_attribute('is_invincible', 'true')
		# set the max speed
		if blueprint.has_attribute('speed'):
			self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
			self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
		else:
			print("No recommended values for 'speed' attribute")


		 # Spawn the player.
		if self.player is not None:
			spawn_point = self.player.get_transform()
			spawn_point.location.z += 2.0
			spawn_point.rotation.roll = 0.0
			spawn_point.rotation.pitch = 0.0
			self.destroy()
			self.player = self.world.try_spawn_actor(blueprint, spawn_point)
			self.show_vehicle_telemetry = False
			self.modify_vehicle_physics(self.player)
		while self.player is None:
			if not self.map.get_spawn_points():
				print('There are no spawn points available in your map/town.')
				print('Please add some Vehicle Spawn Point to your UE4 scene.')
				sys.exit(1)
			spawn_points = self.map.get_spawn_points()
			spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
			spawn_point = carla.Transform(carla.Location(x=-188.1, y=18.5, z=20))
			# spawn_point = carla.Transform()
			
			self.player = self.world.try_spawn_actor(blueprint, spawn_point)
			self.show_vehicle_telemetry = False
			self.modify_vehicle_physics(self.player)
   
		# Set up the sensors.
		self.sensor_manager = SensorManager(self.player, self.sensor_spec)

		actors = self.world.get_actors().filter('traffic.traffic_light*')
		for l in actors:
			l.set_state(carla.TrafficLightState.Green)

	def modify_vehicle_physics(self, actor):
		#If actor is not a vehicle, we cannot use the physics control
		try: 
			physics_control = actor.get_physics_control()
			physics_control.use_sweep_wheel_collision = True
			actor.apply_physics_control(physics_control)
		except Exception:
			pass
  
	def tick(self, clock, frame, display):
		end = self.hud.tick(self, clock, None, frame, display)
		return end

	def render(self, display, frame):
		image = self.sensor_manager.get_data(frame, 'image')
		image = image[:, :, :3]
		image = image[:, :, ::-1]

		# render the view shown in monitor
		if self.rend:
			self.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
			if self.surface is not None:
					display.blit(self.surface, (0, 0))
		return image

	def destroy_sensors(self):
		pass

	def destroy(self):        
		self.sensor_manager.destroy()
		if self.player is not None:
			self.player.destroy()


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
	def __init__(self, width, height, distance=25.0, town='Town05', v_id=1):
		self.dim = (width, height)
		# font = pygame.font.Font(pygame.font.get_default_font(), 20)
		font_name = 'courier' if os.name == 'nt' else 'mono'
		fonts = [x for x in pygame.font.get_fonts() if font_name in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		# mono = pygame.font.match_font(mono)
		# self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
		# self._notifications = FadingText(font, (width, 40), (0, height - 40))
		# self.help = HelpText(pygame.font.Font(mono, 16), width, height)
		self.server_fps = 0
		self.frame = 0
		self.simulation_time = 0
		self._show_info = False
		self._info_text = []
		self._server_clock = pygame.time.Clock()
		self.recording = False
		self.recording_frame = 0
		self.v_id = int(v_id)
		self.ego_data = {}

		self.d_2_intersection = distance
		self.d_last = distance
		self.jam = 0
		
		self.ss_front = []
		self.ss_left = []
		self.ss_right = []
		self.ss_rear = []
		self.ss_rear_left = []
		self.ss_rear_right = []
		
		self.depth_front = []
		self.depth_left = []
		self.depth_right = []
		self.depth_rear = []
		self.depth_rear_left = []
		self.depth_rear_right = []
		self.counter = 0


		self.data_list = []
		self.frame_list = []
		self.sensor_data_list = []
		self.id_list = []
		self.ego_list = []
		
		self.record_flag = False
		self.server_avg_fps = 0

	def on_world_tick(self, timestamp):
		self._server_clock.tick()
		self.server_fps = self._server_clock.get_fps()
		self.server_avg_fps = 0.98 * self.server_avg_fps + 0.02 * self.server_fps
		self.frame = timestamp.frame
		self.simulation_time = timestamp.elapsed_seconds
	
											 
	def tick(self, world, clock, camera, frame, display):
		# self._notifications.tick(world, clock)
		if not self._show_info:
			return
		t = world.player.get_transform()
		v = world.player.get_velocity()
		c = world.player.get_control()
		# print("vehicle height", t)

		compass = world.imu_sensor.compass
		heading = 'N' if compass > 270.5 or compass < 89.5 else ''
		heading += 'S' if 90.5 < compass < 269.5 else ''
		heading += 'E' if 0.5 < compass < 179.5 else ''
		heading += 'W' if 180.5 < compass < 359.5 else ''
		self._info_text = [
			'Server:  % 16.0f FPS' % self.server_fps,
			'Client:  % 16.0f FPS' % clock.get_fps(),
			'',
			'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
			'Map:     % 20s' % world.map.name,
			'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
			'',
			'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
			u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
			'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
			'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
			'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
			'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
			'Height:  % 18.0f m' % t.location.z,
			'']
		if isinstance(c, carla.VehicleControl):
			self._info_text += [
				('Throttle:', c.throttle, 0.0, 1.0),
				('Steer:', c.steer, -1.0, 1.0),
				('Brake:', c.brake, 0.0, 1.0),
				('Reverse:', c.reverse),
				('Hand brake:', c.hand_brake),
				('Manual:', c.manual_gear_shift),
				'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
		elif isinstance(c, carla.WalkerControl):
			self._info_text += [
				('Speed:', c.speed, 0.0, 5.556),
				('Jump:', c.jump)]

		moving = False
		# acc = world.player.get_acceleration().length()

		if c.throttle == 0:
			self.jam += 1
			# print(acc)
			if self.jam > 100:
				return 0
		else:
			self.jam = 0 

		return 1

	def toggle_info(self):
		self._show_info = not self._show_info

	def notification(self, text, seconds=2.0):
		pass
		# self._notifications.set_text(text, seconds=seconds)
		
	def error(self, text):
		pass
		# self._notifications.set_text('Error: %s' % text, (255, 0, 0))


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
# class FadingText(object):
# 	def __init__(self, font, dim, pos):
# 		self.font = font
# 		self.dim = dim
# 		self.pos = pos
# 		self.seconds_left = 0
# 		# self.surface = pygame.Surface(self.dim)

# 	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
# 		text_texture = self.font.render(text, True, color)
# 		# self.surface = pygame.Surface(self.dim)
# 		self.seconds_left = seconds
# 		self.surface.fill((0, 0, 0, 0))
# 		self.surface.blit(text_texture, (10, 11))

# 	def tick(self, _, clock):
# 		delta_seconds = 1e-3 * clock.get_time()
# 		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
# 		self.surface.set_alpha(500.0 * self.seconds_left)

# 	def render(self, display):
# 		display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
# class HelpText(object):
# 	"""Helper class to handle text output using pygame"""

# 	def __init__(self, font, width, height):
# 		lines = __doc__.split('\n')
# 		self.font = font
# 		self.line_space = 18
# 		self.dim = (780, len(lines) * self.line_space + 12)
# 		self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
# 		self.seconds_left = 0
# 		# self.surface = pygame.Surface(self.dim)
# 		self.surface.fill((0, 0, 0, 0))
# 		for n, line in enumerate(lines):
# 			text_texture = self.font.render(line, True, (255, 255, 255))
# 			# self.surface.blit(text_texture, (22, n * self.line_space))
# 			self._render = False
# 		self.surface.set_alpha(220)

# 	def toggle(self):
# 		self._render = not self._render

# 	def render(self, display):
# 		if self._render:
# 			display.blit(self.surface, self.pos)

			
def get_actor_blueprints(world, filter, generation):
	bps = world.get_blueprint_library().filter(filter)

	if generation.lower() == "all":
		return bps

	# If the filter returns only one bp, we assume that this one needed
	# and therefore, we ignore the generation
	if len(bps) == 1:
		return bps

	try:
		int_generation = int(generation)
		# Check if generation is in available generations
		if int_generation in [1, 2]:
			bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
			return bps
		else:
			print("   Warning! Actor Generation is not valid. No actor will be spawned.")
			return []
	except:
		print("   Warning! Actor Generation is not valid. No actor will be spawned.")
		return []
		

def check_close(ev_loc, loc0, distance = 3):
	if ev_loc.distance(loc0) < distance:
		return True

def init_multi_agent(args, world, planner, scenario, roach_policy=None):
	map = world.get_map()
	town = map.name.split('/')[-1]
	agent_list = [scenario['ego_agent']]
	agent_list.extend(scenario['other_agents'])
	sensor_agent_list = []
	interactive_agent_list = []

	for i, agent in enumerate(agent_list):
		agent_dict = {}
		spawn_trans = carla.Transform(carla.Location(agent['start'][0], agent['start'][1]))
		spawn_trans.location.z += 2.0
		spawn_trans.rotation.roll = 0.0
		spawn_trans.rotation.pitch = 0.0
  
		spawn_trans.rotation.yaw = agent['yaw']
   
		####################  set up the locations of destinations, the locations will later be used to calculate A* routes by planner ################
		dest_trans = carla.Location(agent['dest'][0], agent['dest'][1])
		
		# get blueprint from world
		blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
		if blueprint.has_attribute('color'):
			color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', color)
		if blueprint.has_attribute('driver_id'):
			driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
			blueprint.set_attribute('driver_id', driver_id)

		try:
			carla_agent = world.spawn_actor(blueprint, spawn_trans)
			agent_dict['id'] = carla_agent.id
			agent_dict["agent"] = carla_agent
			agent_dict['dest'] = dest_trans
			if "route" in agent:
				waypoints = np.load(f"./route/{town}/{agent['route']}.npy")
				start_index = np.argmin(np.linalg.norm(waypoints - agent['start'], axis=1))
				dest_index = np.argmin(np.linalg.norm(waypoints - agent['dest'], axis=1))
				waypoints = waypoints[start_index:dest_index+1]
				route = get_proxy_route(waypoints, agent['route'])
			else:
				route = planner.trace_route(carla_agent.get_location(), dest_trans)
			agent_dict['route'] = route                    
			agent_dict['done'] = 0
			
		except:
			print("Spawn failed because of collision at spawn position")
		
  
		if i == 0:
			agent_dict['collision'] = CollisionSensor(carla_agent)
  
		if agent['type'] == "sensor":
			# Load agent
			module_name = os.path.basename(args.sensor_agent).split('.')[0]
			sys.path.insert(0, os.path.dirname(args.sensor_agent))
			module_agent = importlib.import_module(module_name)
			agent_class_name = getattr(module_agent, 'get_entry_point')()
			sensor_agent = getattr(module_agent, agent_class_name)(actor=carla_agent, path_to_config=args.agent_config)
   
			agent_dict['model'] = sensor_agent
			agent_dict['name'] = 'sensor'
			sensor_spec_list = sensor_agent.sensors()
			agent_dict['sensors'] = SensorManager(carla_agent, sensor_spec_list)
			sensor_agent_list.append(agent_dict)
		else:
			if agent['type'] == 'roach': 
				# Initialize roach agent
				roach_agent = BEV_MAP(town)
				roach_agent.init_vehicle_bbox(carla_agent.id)
				roach_agent.set_policy(roach_policy)
				agent_dict['model'] = roach_agent
				agent_dict['name'] = 'roach'
			if agent['type'] == "plant":
				plant_agent = PlanTAgent(carla_agent, world, args.plant_config)
				agent_dict['model'] = plant_agent   
				agent_dict['name'] = 'plant' 
			if agent['type'] == "auto":
				auto_agent = AutoPilot(carla_agent, world, route)
				agent_dict['model'] = auto_agent   
				agent_dict['name'] = 'auto' 
			
			interactive_agent_list.append(agent_dict)                

	
	return sensor_agent_list, interactive_agent_list


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
	f = open(args.eval_config)
	eval_config = json.load(f)
	checkpoint = parse_checkpoint(args.checkpoint, args.resume)
	num_scenarios = len(eval_config['available_scenarios'])
   
	if checkpoint['progress'] and bool(args.resume):
		resume = checkpoint['progress'][0]
		if resume == num_scenarios:
			return
	else:
		checkpoint['progress'] = [0, num_scenarios]
		resume = 0
  
	save_path = os.environ.get('SAVE_PATH', None)
	Path(save_path).mkdir(parents=True, exist_ok=True)
	if args.record:
		os.makedirs(f'{save_path}/gifs', exist_ok=True)

	show = os.environ.get("REND", 0)
	for scenario in eval_config["available_scenarios"][resume:]:
		scenario_index = scenario["Index"]
		town = scenario['Town']
		logging.info(f'Running scenario {scenario_index} at {town}')

		client = carla.Client(args.host, args.port)
		client.reload_world()
		client.set_timeout(10.0)

		# Initialize pygame
		if int(show) > 0:
			print("display")
			pygame.init()
			pygame.font.init()
			display = pygame.display.set_mode(
				(512, 512),
				pygame.HWSURFACE | pygame.DOUBLEBUF)
			display.fill((0,0,0))
			pygame.display.flip()
		else:
			display = None
		
		hud = HUD(args.width, args.height, args.distance, town)
		world = World(client.load_world(town), hud, args, scenario['center'])
		avg_FPS = 0
		clock = pygame.time.Clock()

		# spawn other agent 
		map = world.world.get_map()
		spawn_points = map.get_spawn_points()
		planner = GlobalRoutePlanner(map, sampling_resolution=1.0)                

		# Initialize a global roach for all roach agent to avoid generating multipile HD maps
		global_roach = BEV_MAP(town)
		global_roach.init_vehicle_bbox(world.player.id)
		global_roach_policy = global_roach.init_policy()

		sensor_agent_list, interactive_agent_list = init_multi_agent(args, world.world, planner, scenario, global_roach_policy)
		start_frame = None

		if save_path and args.record:
			image_buffer = []

		while True:
			clock.tick_busy_loop(30)
			frame = world.world.tick()
			if not start_frame:
				start_frame = frame
			
			if display:
				view = pygame.surfarray.array3d(display)
				view = view.transpose([1, 0, 2]) 
				image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)        
			else:
				image = None
    
			world.tick(clock, frame, image)
			avg_FPS = 0.98 * avg_FPS + 0.02 * clock.get_fps()
			
			# Set traffic lights to green
			traffic_light_actors = world.world.get_actors().filter('traffic.traffic_light*')
			for l in traffic_light_actors:
				l.set_state(carla.TrafficLightState.Green)
		
			# continue
			# collect data for all roach if needed
			if global_roach:
				processed_data = global_roach.collect_actor_data(world)
			
			all_agent_list = interactive_agent_list + sensor_agent_list
			################### Check every agent route and generate new one if needed. #####################
			for agent_dict in all_agent_list:                        
				# regenerate a route when the agent deviates from the current route
				if not check_close(agent_dict["agent"].get_location(), agent_dict['route'][0][0].transform.location, 6):
					logging.debug(f"route deviation: {agent_dict['name']}_{agent_dict['id']}")
					destination = agent_dict['route'][-1][0].transform.location
					agent_dict['route'] = planner.trace_route(agent_dict["agent"].get_location(), destination)
				
				# Delete reached points from current route
				while check_close(agent_dict["agent"].get_location(), agent_dict['route'][0][0].transform.location):
					agent_dict['route'].pop(0)
					if len(agent_dict['route']) == 0:
						logging.info(f"route complete: {agent_dict['name']}_{agent_dict['id']}")
						agent_dict['done'] = 1
						new_destination = random.choice(spawn_points).location
						agent_dict['route'] = planner.trace_route(agent_dict["agent"].get_location(), new_destination)

			################### Prepare input for each agent. #####################
			t_list = []
			for agent_dict in all_agent_list:
				if agent_dict['name'] == 'roach':
					route_list = [wp[0].transform.location for wp in agent_dict['route'][0:60]]
					if args.debug:
						for w in route_list:
							world.world.debug.draw_string(w, 'O', draw_shadow=False,
														color=carla.Color(r=255, g=0, b=0), life_time=10.0,
														persistent_lines=True)
					inputs = [route_list, agent_dict]
					agent_dict['model'].set_data(processed_data)

				if agent_dict['name'] == 'plant':
					route_list = [wp[0].transform.location for wp in agent_dict['route'][0:60]]
					if args.debug:
						for w in route_list:
							world.world.debug.draw_string(w, 'O', draw_shadow=False,
														color=carla.Color(r=255, g=0, b=0), life_time=10.0,
														persistent_lines=True)
					tick_data = agent_dict['model'].tick(agent_dict)
					inputs = [tick_data, agent_dict]
				
				if agent_dict['name'] == 'sensor':
					route_list = [wp for wp in agent_dict['route'][0:60]]
					if args.debug:
						for w, _ in route_list:
							world.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
														color=carla.Color(r=255, g=0, b=0), life_time=10.0,
														persistent_lines=True)

					rgb = agent_dict['sensors'].get_data(frame, 'image')
					lidar = agent_dict['sensors'].get_data(frame, 'lidar')
					tick_data = agent_dict['model'].tick(rgb, lidar, agent_dict)
					inputs = [tick_data, agent_dict]

				if agent_dict['name'] == "auto":
					route_list = [wp for wp in agent_dict['route'][0:60]]
					if args.debug:
						for w, _ in route_list:
							world.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
														color=carla.Color(r=255, g=0, b=0), life_time=10.0,
														persistent_lines=True)
					agent_dict['model'].tick(agent_dict)
					inputs = [route_list, agent_dict]


				t = Thread(target=agent_dict['model'].run_step, args=tuple(inputs))
				t_list.append(t)
				t.start()
			
			for t in t_list:
				t.join()
    
			################### Apply control to each agent. #####################
			for agent_dict in all_agent_list:
				control = agent_dict["control"] 
				agent_dict["agent"].apply_control(control)

			################### Render scene in both sever and client. #####################
			bev_image = world.render(display, frame)
			if display:
				pygame.display.flip()
			if save_path and args.record and frame%5 == 0:
				image_buffer.append(Image.fromarray(bev_image))

			################### Check if all agent complete the scenario. #####################
			scene_done = 1
			for agent_dict in all_agent_list:
				if agent_dict['name'] == 'sensor':
					scene_done = scene_done and agent_dict['done']
			if scene_done:
				fps = (frame - start_frame)/hud.simulation_time
				finish_time = hud.simulation_time
				break
			
			################### Break if timeout. #####################
			if hud.simulation_time > 30:
				logging.info('simulation timeout after 30 second')
				fps = (frame - start_frame)/hud.simulation_time
				finish_time = hud.simulation_time
				break
		
		logging.debug(f"FPS: {fps}")
	
		################### Compute number of collision. #####################
		collision_count = 0
		for agent_dict in all_agent_list:
			id_record = {}
			if 'collision' in agent_dict:
				for idx, frame in enumerate(agent_dict['collision'].frame_history):
					ids = agent_dict['collision'].id_history[idx]
					for id in ids:
						if id in id_record and (frame - id_record[id])/fps > 1:
							collision_count += 1
						id_record[id] = frame
				collision_count += len(id_record)
	
		logging.info(f"Simulation time: {finish_time}")			
		logging.debug("Destroy env")
		
		for agent_dict in all_agent_list:
			if "collision" in agent_dict:
				agent_dict['collision'].sensor.stop()
				agent_dict['collision'].sensor.destroy()
			if 'sensors' in agent_dict:
				agent_dict['sensors'].destroy()
			agent_dict['model'].destroy()


		client.apply_batch([carla.command.DestroyActor(x['agent']) for x in all_agent_list])
		world.destroy()
		del all_agent_list
		del sensor_agent_list
		del interactive_agent_list                
		del client
		del world
		del hud
		del global_roach
		del global_roach_policy
		
		if collision_count > 0 or finish_time > 30:
			success = False
		else:
			success = True

		result = {
			"Index": scenario_index,
			"Collisions": collision_count,
			"Completion Time": finish_time,
			"Success": success,
		}

		checkpoint['records'].append(result)
		checkpoint['progress'][0] = scenario_index + 1
		with open(args.checkpoint, 'w') as fd:
			json.dump(checkpoint, fd, indent=2, sort_keys=True)
		
		if len(image_buffer) > 0:
			image_buffer[0].save(
				f'{save_path}/gifs/{scenario_index}.gif',
				save_all=True,
        append_images=image_buffer[1:],
        duration=int(0.0005 * 1000),
        loop=0,  # 0 means an infinite loop
			)
		
		if display:
			pygame.quit()
		logging.debug(f"Finish scenario{scenario_index}")
  
	avg_completion_time = 0
	success_rate = 0
	collision_rate = 0
	for record in checkpoint['records']:
		avg_completion_time += record['Completion Time']
		success_rate += record['Success']
		collision_rate += record['Collisions']
	print(success_rate)
	print(num_scenarios)
	avg_completion_time /= num_scenarios
	success_rate /= num_scenarios
	collision_rate /= num_scenarios
	checkpoint['global record'] = {
		"Avg Completion Time": round(avg_completion_time, 2),
		"Success Rate": round(success_rate*100, 2),
		"Collision Rate": round(collision_rate, 2)
	}

	with open(args.checkpoint, 'w') as fd:
			json.dump(checkpoint, fd, indent=2, sort_keys=True)
   

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
	argparser = argparse.ArgumentParser(
		description='Interaction Benchmark')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'--res',
		metavar='WIDTHxHEIGHT',
		default='512x512',
		help='window resolution (default: 1280x720)')
	argparser.add_argument(
		'--filter',
		metavar='PATTERN',
		default='model3',
		help='actor filter (default: "vehicle.*")')
	argparser.add_argument(
		'--rolename',
		metavar='NAME',
		default='hero',
		help='actor role name (default: "hero")')
	argparser.add_argument(
		'--distance',
		default=25.0,
		type=float,
		help='distance to intersection for toggling camera)')
	argparser.add_argument(
		'--eval_config',
		default='eval_config.json',
		type=str,
		help='Path to evaluation config')
	argparser.add_argument(
		'--sensor_agent',
		default='roach',
		type=str,
		help='Path ego agent entry')
	argparser.add_argument(
		'--agent_config',
		type=str,
		help='Path to ego agent config')
	argparser.add_argument(
		'--plant_config',
		default='checkpoints/PlanT',
		type=str,
		help='Path to plant agent config if needed')
	argparser.add_argument(
   	'--debug',
		default=False,
		type=bool,
		help='visualize debug information')
	argparser.add_argument(
   	'--checkpoint',
		default='results.json',
		type=str,
		help='Path to checkpoint used for saving statistics and resuming')
	argparser.add_argument(
   	'--resume',
		default='1',
		type=int,
		help='Weather to resume the evaluation from checkpoint, default is set to True')
	argparser.add_argument(
   	'--record',
		default='1',
		type=int,
		help='Weather to record video and save in SAVE_PATH/gifs')
	args = argparser.parse_args()

	args.width, args.height = [int(x) for x in args.res.split('x')]

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	# print(__doc__)

	try:
		game_loop(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')

if __name__ == '__main__':

	main()