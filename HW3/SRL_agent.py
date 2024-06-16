import os
from copy import deepcopy

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math

from team_code.model import LidarCenterNet
from team_code.config import GlobalConfig
from team_code.data import CARLA_Data

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

import team_code.transfuser_utils as t_u

import pickle
from copy import deepcopy

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

def get_entry_point():
  return 'SRLAgent'

class SRLAgent():
  def __init__(self, actor, path_to_config) -> None:
    torch.cuda.empty_cache()
    self.config_path = path_to_config
    self.device = torch.device('cuda:0')
    self.step = 0
    self.actor = actor
    
    # Load the config saved during training
    with open(os.path.join(self.config_path, 'config.pickle'), 'rb') as args_file:
      loaded_config = pickle.load(args_file)
    
    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)
    
    
    # For models supporting different output modalities we select which one to use here.
    # 0: Waypoints
    # 1: Path + Target Speed
    direct = os.environ.get('DIRECT', 1)
    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    if direct is not None:
      self.config.inference_direct_controller = int(direct)
      print('Direct control prediction?: ', direct)

    # If set to true, will generate visualizations at SAVE_PATH
    self.config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0)) == 1

    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))

    # Classification networks are known to be overconfident which leads to them braking a bit too late in our case.
    # Reducing the driving speed slightly counteracts that.
    if int(os.environ.get('SLOWER', 1)):
      print('Reduce target speed value by one.')
      self.config.target_speeds[2] = self.config.target_speeds[2] - 1.0
      self.config.target_speeds[3] = self.config.target_speeds[3] - 1.0

    # Collects some statistics about the target point. Not needed usually.
    self.tp_stats = False
    self.tp_sign_agrees_with_angle = []
    if int(os.environ.get('TP_STATS', 0)):
      self.tp_stats = True

    if self.config.tp_attention:
      self.tp_attention_buffer = []

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(self.config_path):
      if file.endswith('.pth'):
        self.model_count += 1
        print(os.path.join(self.config_path, file))
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(self.config_path, file), map_location=self.device)

        net.load_state_dict(state_dict, strict=False)
        net.cuda(device=self.device)
        net.eval()
        self.nets.append(net)

    self.stuck_detector = 0
    self.force_move = 0

    self.bb_buffer = deque(maxlen=1)
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.target_point_prev = [1e5, 1e5]

    # Filtering
    self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
    self.ukf = UKF(dim_x=4,
                   dim_z=4,
                   fx=bicycle_model_forward,
                   hx=measurement_function_hx,
                   dt=self.config.carla_frame_rate,
                   points=self.points,
                   x_mean_fn=state_mean,
                   z_mean_fn=measurement_mean,
                   residual_x=residual_state_x,
                   residual_z=residual_measurement_h)

    # State noise, same as measurement because we
    # initialize with the first measurement later
    self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
    # Measurement noise
    self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
    self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
    # Used to set the filter state equal the first measurement
    self.filter_initialized = False
    # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
    self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

    #Temporal LiDAR
    self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
    self.lidar_last = None
    self.initialized = False
    control = carla.VehicleControl(steer=0.0, throttle=0.3, brake=0.0)
    self.control = control
    
    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)
    direct = os.environ.get('DIRECT', 1)
    if direct is not None:
      self.config.inference_direct_controller = int(direct)
      print('Direct control prediction?: ', direct)
  
  def destroy(self):
    del self.tp_sign_agrees_with_angle
    del self.nets
    del self.config
  
  def sensors(self):
    sensors = [{
        'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'id': 'lidar'
    }, {
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_front'
    }]

    # sensors = [{
    #     'type': 'sensor.camera.rgb',
    #     'x': self.config.camera_pos[0],
    #     'y': self.config.camera_pos[1],
    #     'z': self.config.camera_pos[2],
    #     'roll': self.config.camera_rot_0[0],
    #     'pitch': self.config.camera_rot_0[1],
    #     'yaw': self.config.camera_rot_0[2],
    #     'width': self.config.camera_width,
    #     'height': self.config.camera_height,
    #     'fov': self.config.camera_fov,
    #     'id': 'rgb_front'
    # }]
    #       'type': 'sensor.lidar.ray_cast',
    #       'x': self.config.lidar_pos[0],
    #       'y': self.config.lidar_pos[1],
    #       'z': self.config.lidar_pos[2],
    #       'roll': self.config.lidar_rot[0],
    #       'pitch': self.config.lidar_rot[1],
    #       'yaw': self.config.lidar_rot[2],
    #       'id': 'lidar'
    # }]

    return sensors
  
  @torch.inference_mode()  # Turns off gradient computation
  def tick(self, image, lidar, input_data):    
    location = self.actor.get_location()
    pos = np.array([location.x, location.y])

    compass = np.deg2rad(self.actor.get_transform().rotation.yaw)
    speed = self._get_forward_speed()
    
    rgb = []
    
    if image is not None:
      camera = image[:, :, :3]

      # Also add jpg artifacts at test time, because the training data was saved as jpg.
      _, compressed_image_i = cv2.imencode('.jpg', camera)
      camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)

      rgb = np.concatenate(rgb, axis=1)
      rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)


    result = {
        'rgb': rgb,
        'compass': compass,
    }

    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)
  
    if lidar is not None:
      result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, lidar)

    if not self.filter_initialized:
      self.ukf.x = np.array([pos[0], pos[1], t_u.normalize_angle(compass), speed])
      self.filter_initialized = True

    self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    self.ukf.update(np.array([pos[0], pos[1], t_u.normalize_angle(compass), speed]))
    filtered_state = self.ukf.x
    self.state_log.append(filtered_state)

    result['gps'] = filtered_state[0:2]

    route_list = [wp for wp in input_data['route'][0:60]]
    
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

    ego_target_point = t_u.inverse_conversion_2d(target_point, result['gps'], result['compass'])
    ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    result['target_point'] = ego_target_point
    
    return result
  
  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, tick_data, input_data):
    self.step += 1
    # tick_data = self.tick(input_data)
    
    if not self.initialized:
      self.initialized = True
      control = carla.VehicleControl(steer=0.0, throttle=0.3, brake=0.0)
      self.control = control
      if "lidar" in tick_data:
        self.lidar_last = deepcopy(tick_data['lidar'])
      input_data['control'] = control
      return

    
    lidar_indices = []
    for i in range(self.config.lidar_seq_len):
      lidar_indices.append((i + 1) * self.config.data_save_freq - 1)

    #Current position of the car
    ego_x = self.state_log[-1][0]
    ego_y = self.state_log[-1][1]
    ego_theta = self.state_log[-1][2]

    ego_x_last = self.state_log[-2][0]
    ego_y_last = self.state_log[-2][1]
    ego_theta_last = self.state_log[-2][2]

    lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    lidar_current = deepcopy(tick_data['lidar'])
    lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

    self.lidar_buffer.append(lidar_full)

      # We wait until we have sufficient LiDARs
    if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
      self.lidar_last = deepcopy(tick_data['lidar'])
      tmp_control = carla.VehicleControl(0.0, 0.3, 0.0)
      self.control = tmp_control

      input_data['control'] = tmp_control
      return

    # Possible action repeat configuration
    if self.step % self.config.action_repeat == 1:
      self.lidar_last = deepcopy(tick_data['lidar'])

      input_data['control'] = self.control
      return
    
    # Voxelize LiDAR and stack temporal frames
    lidar_bev = []
    # prepare LiDAR input
    for i in lidar_indices:
      lidar_point_cloud = deepcopy(self.lidar_buffer[i])

      # For single frame there is no point in realignment. The state_log index will also differ.
      if self.config.realign_lidar and self.config.lidar_seq_len > 1:
        # Position of the car when the LiDAR was collected
        curr_x = self.state_log[i][0]
        curr_y = self.state_log[i][1]
        curr_theta = self.state_log[i][2]

        # Voxelize to BEV for NN to process
        lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

      lidar_histogram = torch.from_numpy(
          self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                use_ground_plane=self.config.use_ground_plane)).unsqueeze(0)

      lidar_histogram = lidar_histogram.to(self.device, dtype=torch.float32)
      lidar_bev.append(lidar_histogram)

    lidar_bev = torch.cat(lidar_bev, dim=1)

    self.lidar_last = deepcopy(tick_data['lidar'])

    # prepare velocity input
    gt_velocity = tick_data['speed']
    velocity = gt_velocity.reshape(1, 1)  # used by transfuser

    # forward pass
    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    bounding_boxes = []
    wp_selected = None
    for i in range(self.model_count):
      pred_wp,\
      pred_target_speed,\
      pred_checkpoint,\
      pred_semantic, \
      pred_bev_semantic, \
      pred_bev_topo, \
      pred_depth, \
      pred_hdmap, \
      pred_route, \
      pred_traj, \
      pred_vehicle_occupancy, \
      pred_pedestrian_occupancy, \
      pred_bounding_box, pred_occ, attention_weights, \
      pred_wp_1, \
      selected_path = self.nets[i].forward(
        rgb=tick_data['rgb'],
        lidar_bev=lidar_bev,
        target_point=tick_data['target_point'],
        ego_vel=velocity,
        command=tick_data['command'])
      
      if self.config.use_wp_gru:
        if self.config.multi_wp_output:
          wp_selected = 0
          if F.sigmoid(selected_path)[0].item() > 0.5:
            wp_selected = 1
            pred_wps.append(pred_wp_1)
          else:
            pred_wps.append(pred_wp)
        else:
          pred_wps.append(pred_wp)
      if self.config.use_controller_input_prediction:
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0][1])


    # if self.stop_sign_controller:
    #   stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    if self.config.use_controller_input_prediction:
      pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
      pred_aim_wp = torch.stack(pred_checkpoints, dim=0).mean(dim=0)

      pred_aim_wp = pred_aim_wp.detach().cpu().numpy()
      pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

      if self.tp_stats:
        loc_tp = tick_data['target_point'].detach().cpu().numpy()[0]
        deg_pred_angle = pred_angle * 90.0
        tp_angle = -math.degrees(math.atan2(-loc_tp[1], loc_tp[0]))
        if abs(tp_angle) > 1.0 and abs(deg_pred_angle) > 1.0:
          same_direction = float(tp_angle * deg_pred_angle >= 0.0)
          self.tp_sign_agrees_with_angle.append(same_direction)

      if self.uncertainty_weight:
        uncertainty = pred_target_speed.detach().cpu().numpy()
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed = self.config.target_speeds[0]
        else:
          pred_target_speed = sum(uncertainty * self.config.target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed)
        pred_target_speed = self.config.target_speeds[pred_target_speed_index]

    if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_target_speed, pred_angle, gt_velocity)
    elif self.config.use_wp_gru and not self.config.inference_direct_controller:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity)
    else:
      raise ValueError('An output representation was chosen that was not trained.')

    # 0.1 is just an arbitrary low number to threshold when the car is stopped
    if gt_velocity < 0.1:
      self.stuck_detector += 1
    else:
      self.stuck_detector = 0

    # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
    if self.stuck_detector > self.config.stuck_threshold:
      self.force_move = self.config.creep_duration

    if self.force_move > 0:
      emergency_stop = False
      if self.config.backbone not in ('aim'):
        # safety check
        safety_box = deepcopy(self.lidar_buffer[-1])

        # z-axis
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

        # y-axis
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

        # x-axis
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

      if not emergency_stop:
        print('Detected agent being stuck. Step: ', self.step)
        throttle = max(self.config.creep_throttle, throttle)
        brake = False
        self.force_move -= 1
      else:
        print('Creeping stopped by safety box. Step: ', self.step)
        throttle = 0.0
        brake = True
        self.force_move = self.config.creep_duration

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

    # CARLA will not let the car drive in the initial frames.
    # We set the action to brake so that the filter does not get confused.
    if self.step < self.config.inital_frames_delay:
      self.control = carla.VehicleControl(0.0, 0.3, 0.0)
    else:
      self.control = control

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
  
  def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar(lidar, pos_diff, rot_diff)

  def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target]) - np.array([x, y])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                                [np.sin(orientation_target), np.cos(orientation_target)]])
    pos_diff = rotation_matrix.T @ pos_diff

    # Rotation matrix in local coordinate system
    local_rot_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff)], [np.sin(rot_diff), np.cos(rot_diff)]])

    for _, box_pred in enumerate(boxes):
      box_pred[:2] = (local_rot_matrix.T @ (box_pred[:2] - pos_diff).T).T
      box_pred[4] = t_u.normalize_angle(box_pred[4] - rot_diff)


  
  
  
# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
  # Kinematic bicycle model.
  # Numbers are the tuned parameters from World on Rails
  front_wb = -0.090769015
  rear_wb = 1.4178275

  steer_gain = 0.36848336
  brake_accel = -4.952399
  throt_accel = 0.5633837

  locs_0 = x[0]
  locs_1 = x[1]
  yaw = x[2]
  speed = x[3]

  if brake:
    accel = brake_accel
  else:
    accel = throt_accel * throttle

  wheel = steer_gain * steer

  beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
  next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
  next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
  next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
  next_speed = speed + accel * dt
  next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

  next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

  return next_state_x


def measurement_function_hx(vehicle_state):
  '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
  return vehicle_state


def state_mean(state, wm):
  '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def measurement_mean(state, wm):
  '''
  We use the arctan of the average of sin and cos of the angle to
  calculate the average of orientations.
  :param state: array of states to be averaged. First index is the
  timestep.
  '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def residual_state_x(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


def residual_measurement_h(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y