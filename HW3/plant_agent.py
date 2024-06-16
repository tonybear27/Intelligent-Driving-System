import os
import torch
import torch.nn.functional as F
import pickle
from team_code.plant import PlanT
from team_code.data import CARLA_Data
import math
from collections import deque
import numpy as np

import carla
from team_code.config import GlobalConfig
import team_code.transfuser_utils as t_u

class PlanTAgent():
	def __init__(self, actor, world, path_to_conf_file):
		torch.cuda.empty_cache()

		with open(os.path.join(path_to_conf_file, 'config.pickle'), 'rb') as args_file:
			loaded_config = pickle.load(args_file)

		# Generate new config for the case that it has new variables.
		self.config = GlobalConfig()
		# Overwrite all properties that were set in the save config.
		self.config.__dict__.update(loaded_config.__dict__)

		self.config.debug = int(os.environ.get('VISU_PLANT', 0)) == 1
		self.device = torch.device('cuda:0')

		self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

		self.config.inference_direct_controller = 1
		self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
		print('Uncertainty weighting?: ', self.uncertainty_weight)
		self.config.brake_uncertainty_threshold = float(
				os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
		if self.uncertainty_weight:
			print('Uncertainty threshold: ', self.config.brake_uncertainty_threshold)

		# Load model files
		self.nets = []
		self.model_count = 0  # Counts how many models are in our ensemble
		for file in os.listdir(path_to_conf_file):
			if file.endswith('.pth'):
				self.model_count += 1
				print(os.path.join(path_to_conf_file, file))
				net = PlanT(self.config)
				if self.config.sync_batch_norm:
					# Model was trained with Sync. Batch Norm.
					# Need to convert it otherwise parameters will load wrong.
					net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
				state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location=self.device)

				net.load_state_dict(state_dict, strict=False)
				net.cuda()
				net.eval()
				self.nets.append(net)

		if self.config.debug:
			self.init_map = False
			
		self.actor = actor
		self._world = world
		
		self.initialized = False
		self.target_point_prev = [1e5, 1e5]
		self.commands = deque(maxlen=2)
		self.commands.append(4)
		self.commands.append(4)

	def destroy(self):
		del self.nets

		
	def tick(self, input_data):
		result = {}

		location = self.actor.get_location()
		pos = np.array([location.x, location.y])
		
		speed = self._get_forward_speed()
		result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

		compass = np.deg2rad(self.actor.get_transform().rotation.yaw)
		route_list = [wp for wp in input_data['route'][0:60]]
		
		dense_route = []
		if len(route_list) >= self.config.num_route_points_saved:
			remaining_route = list(route_list)[:self.config.num_route_points_saved]
		else:
			remaining_route = list(route_list)

		for checkpoint in remaining_route:
			dense_route.append(t_u.inverse_conversion_2d(np.array([checkpoint[0].transform.location.x, checkpoint[0].transform.location.y]), pos, compass).tolist())	
		result['route'] = dense_route
  
		if len(route_list) > 2:
			target_point, far_command = route_list[1]
		elif len(route_list) > 1:
			target_point, far_command = route_list[1]
		else:
			target_point, far_command = route_list[0]
			
		target_point = np.array([target_point.transform.location.x, target_point.transform.location.y])
		if (target_point != self.target_point_prev).all():
			self.target_point_prev = target_point
			self.commands.append(far_command.value)

		one_hot_command = t_u.command_to_one_hot(self.commands[-2])
		result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

		ego_target_point = t_u.inverse_conversion_2d(target_point, pos, compass)
		ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

		result['target_point'] = ego_target_point  

		# We do not consider traffic light and stop sign in this benchmark
		result['light_hazard'] = False
		result['stop_sign_hazard'] = False    

		ego_vehicle_waypoint = self._world.get_map().get_waypoint(self.actor.get_location())
		result['junction'] = ego_vehicle_waypoint.is_junction
		
		bounding_boxes = self.get_bounding_boxes()
		result['bounding_boxes'] = bounding_boxes
  
		return result
		
	@torch.inference_mode()
	def run_step(self, tick_data, input_data):  # pylint: disable=locally-disabled, unused-argument
		if not self.initialized:
			self.initialized = True
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 1.0
			input_data['control'] = control
			return

		target_point = torch.tensor(tick_data['target_point'], dtype=torch.float32).to(self.device).unsqueeze(0)

		# Preprocess route the same way we did during training
		route = tick_data['route']
		if len(route) < self.config.num_route_points:
			num_missing = self.config.num_route_points - len(route)
			route = np.array(route)
			# Fill the empty spots by repeating the last point.
			route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
		else:
			route = np.array(route[:self.config.num_route_points])
		
		if self.config.smooth_route:
			route = self.data.smooth_path(route)
		route = torch.tensor(route, dtype=torch.float32)[:self.config.num_route_points].to(self.device).unsqueeze(0)

		light_hazard = torch.tensor(tick_data['light_hazard'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
		stop_sign_hazard = torch.tensor(tick_data['stop_sign_hazard'],
																		dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
		junction = torch.tensor(tick_data['junction'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)

		bounding_boxes, _ = self.data.parse_bounding_boxes(tick_data['bounding_boxes'])
		bounding_boxes_padded = torch.zeros((self.config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

		if len(bounding_boxes) > 0:
			# Pad bounding boxes to a fixed number
			bounding_boxes = np.stack(bounding_boxes)
			bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32).to(self.device)

			if bounding_boxes.shape[0] <= self.config.max_num_bbs:
				bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
			else:
				bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]

		bounding_boxes_padded = bounding_boxes_padded.unsqueeze(0)

		speed = torch.tensor(tick_data['speed'], dtype=torch.float32).to(self.device).unsqueeze(0)

		pred_wps = []
		pred_target_speeds = []
		pred_checkpoints = []
		pred_bbs = []
		for i in range(self.model_count):
			pred_wp, pred_target_speed, pred_checkpoint, pred_bb, _ = self.nets[i].forward(bounding_boxes=bounding_boxes_padded,
																																									route=route,
																																									target_point=target_point,
																																									light_hazard=light_hazard,
																																									stop_hazard=stop_sign_hazard,
																																									junction=junction,
																																									velocity=speed)

			pred_wps.append(pred_wp)
			pred_bbs.append(t_u.plant_quant_to_box(self.config, pred_bb))
			if self.config.use_controller_input_prediction:
				pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
				pred_checkpoints.append(pred_checkpoint[0][1])

		if self.config.use_wp_gru:
			self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

		pred_bbs = torch.stack(pred_bbs, dim=0).mean(dim=0)

		if self.config.use_controller_input_prediction:
			pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
			pred_aim_wp = torch.stack(pred_checkpoints, dim=0).mean(dim=0)
			pred_aim_wp = pred_aim_wp.detach().cpu().numpy()
			pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

			if self.uncertainty_weight:
				uncertainty = pred_target_speed.detach().cpu().numpy()
				if uncertainty[0] > self.config.brake_uncertainty_threshold:
					pred_target_speed = self.config.target_speeds[0]
				else:
					pred_target_speed = sum(uncertainty * self.config.target_speeds)
			else:
				pred_target_speed_index = torch.argmax(pred_target_speed)
				pred_target_speed = self.config.target_speeds[pred_target_speed_index]
    
		if self.config.inference_direct_controller and \
				self.config.use_controller_input_prediction:
			steer, throttle, brake = self.nets[0].control_pid_direct(pred_target_speed, pred_angle, speed, False)
		else:
			steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, speed, False)

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)
		input_data['control'] = control


	def _get_forward_speed(self, transform=None, velocity=None):
		""" Convert the vehicle transform directly to forward speed """
		if not velocity:
			velocity = self.actor.get_velocity()
		if not transform:
			transform = self.actor.get_transform()

		vel_np = np.array([velocity.x, velocity.y, velocity.z])
		pitch = np.deg2rad(transform.rotation.pitch)
		yaw = np.deg2rad(transform.rotation.yaw)
		orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
		speed = np.dot(vel_np, orientation)
		return speed
	
	def get_bounding_boxes(self):
		results = []

		ego_transform = self.actor.get_transform()
		ego_control = self.actor.get_control()
		ego_velocity = self.actor.get_velocity()
		ego_matrix = np.array(ego_transform.get_matrix())
		ego_rotation = ego_transform.rotation
		ego_extent = self.actor.bounding_box.extent
		ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
		ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
		ego_yaw = np.deg2rad(ego_rotation.yaw)
		ego_brake = ego_control.brake

		relative_yaw = 0.0
		relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

		result = {
				'class': 'ego_car',
				'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
				'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
				'yaw': relative_yaw,
				'num_points': -1,
				'distance': -1,
				'speed': ego_speed,
				'brake': ego_brake,
				'id': int(self.actor.id),
				'matrix': ego_transform.get_matrix()
		}
		results.append(result)

		self._actors = self._world.get_actors()
		vehicles = self._actors.filter('*vehicle*')

		for vehicle in vehicles:
			if vehicle.get_location().distance(self.actor.get_location()) < self.config.bb_save_radius:
				if vehicle.id != self.actor.id:
					vehicle_transform = vehicle.get_transform()
					vehicle_rotation = vehicle_transform.rotation
					vehicle_matrix = np.array(vehicle_transform.get_matrix())
					vehicle_control = vehicle.get_control()
					vehicle_velocity = vehicle.get_velocity()
					vehicle_extent = vehicle.bounding_box.extent
					vehicle_id = vehicle.id

					vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
					yaw = np.deg2rad(vehicle_rotation.yaw)

					relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
					relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
					vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
					vehicle_brake = vehicle_control.brake

					num_in_bbox_points = -1

					distance = np.linalg.norm(relative_pos)

					result = {
							'class': 'car',
							'extent': vehicle_extent_list,
							'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
							'yaw': relative_yaw,
							'num_points': int(num_in_bbox_points),
							'distance': distance,
							'speed': vehicle_speed,
							'brake': vehicle_brake,
							'id': int(vehicle_id),
							'matrix': vehicle_transform.get_matrix()
					}
					results.append(result)

		walkers = self._actors.filter('*walker*')
		for walker in walkers:
			if walker.get_location().distance(self.actor.get_location()) < self.config.bb_save_radius:
				walker_transform = walker.get_transform()
				walker_velocity = walker.get_velocity()
				walker_rotation = walker.get_transform().rotation
				walker_matrix = np.array(walker_transform.get_matrix())
				walker_id = walker.id
				walker_extent = walker.bounding_box.extent
				walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
				yaw = np.deg2rad(walker_rotation.yaw)

				relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
				relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

				walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

				num_in_bbox_points = -1

				distance = np.linalg.norm(relative_pos)

				result = {
						'class': 'walker',
						'extent': walker_extent,
						'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
						'yaw': relative_yaw,
						'num_points': int(num_in_bbox_points),
						'distance': distance,
						'speed': walker_speed,
						'id': int(walker_id),
						'matrix': walker_transform.get_matrix()
				}
				results.append(result)

		return results