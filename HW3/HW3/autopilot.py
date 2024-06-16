"""
Autopilot for evaluation, modified from https://github.com/autonomousvision/carla_garage
"""

import statistics

import math
import numpy as np
import carla

from collections import defaultdict
from team_code.config import GlobalConfig
import team_code.transfuser_utils as t_u
from team_code.nav_planner import RoutePlanner, PIDController



class AutoPilot():
  def __init__(self, actor, world, route_list):
    self.config = GlobalConfig()

    # Dynamics models
    self.ego_model = EgoModel(dt=(1.0 / self.config.bicycle_frame_rate))
    self.vehicle_model = EgoModel(dt=(1.0 / self.config.bicycle_frame_rate))

    # Controllers
    self._turn_controller = PIDController(k_p=self.config.turn_kp,
                                          k_i=self.config.turn_ki,
                                          k_d=self.config.turn_kd,
                                          n=self.config.turn_n)
    self._turn_controller_extrapolation = PIDController(k_p=self.config.turn_kp,
                                                        k_i=self.config.turn_ki,
                                                        k_d=self.config.turn_kd,
                                                        n=self.config.turn_n)
    self._speed_controller = PIDController(k_p=self.config.speed_kp,
                                           k_i=self.config.speed_ki,
                                           k_d=self.config.speed_kd,
                                           n=self.config.speed_n)
    self._speed_controller_extrapolation = PIDController(k_p=self.config.speed_kp,
                                                         k_i=self.config.speed_ki,
                                                         k_d=self.config.speed_kd,
                                                         n=self.config.speed_n)

    self.list_traffic_lights = []
    
    # Initialize controls
    self.steer = 0.0
    self.throttle = 0.0
    self.brake = 0.0
    self.target_speed = self.config.target_speed_fast

    # Angle to the next waypoint.
    # Normalized in [-1, 1] corresponding to [-90, 90]
    self.angle = 0.0
    self.stop_sign_hazard = False
    self.traffic_light_hazard = False
    self.walker_hazard = False
    self.vehicle_hazard = False
    self.vehicle_ids = []
    self.junction = False
    self.aim_wp = None  # Waypoint that the expert is steering towards
    self.close_traffic_lights = []
    self.close_stop_signs = []
    # A list of all stop signs that we have cleared
    self.cleared_stop_signs = []
    self.visible_walker_ids = []
    self.walker_past_pos = {}  # Position of walker in the last frame
    self.visualize = 0
    
    self.actor = actor
    self._world = world
    
    self._waypoint_planner_extrapolation = RoutePlanner(self.config.dense_route_planner_min_distance,
                                                        self.config.dense_route_planner_max_distance)
    
    self._waypoint_planner_extrapolation.set_route(route_list)
    self._waypoint_planner_extrapolation.save()
    
    # Speed buffer for detecting "stuck" vehicles
    self.vehicle_speed_buffer = defaultdict(lambda: {'velocity': [], 'throttle': [], 'brake': []})    

  def destroy(self):
    del self.visible_walker_ids
    del self.walker_past_pos
    del self._waypoint_planner_extrapolation
    del self.ego_model
    del self._turn_controller
    del self._turn_controller_extrapolation
    del self._speed_controller
    del self._speed_controller_extrapolation
    del self.actor
    del self._world

  def tick(self, input_data):
    compass = np.deg2rad(self.actor.get_transform().rotation.yaw)
    input_data['compass'] = compass
    input_data['speed'] = self._get_forward_speed()


  def run_step(self, route_list, input_data):
    location = input_data['agent'].get_location()
    pos = np.array([location.x, location.y])
    
    brake = self._get_brake()

    ego_vehicle_waypoint = self._world.get_map().get_waypoint(input_data['agent'].get_location())
    self.junction = ego_vehicle_waypoint.is_junction

    speed = input_data['speed']
    if self.walker_close or self.stop_sign_close:
      target_speed = self.config.target_speed_walker
    elif self.junction:
      target_speed = self.config.target_speed_slow
    else:
      target_speed = self.config.target_speed_fast

    # Update saved route
    self._waypoint_planner_extrapolation.run_step(pos)
    self._waypoint_planner_extrapolation.save()

    # Should brake represents braking due to control
    throttle, control_brake = self._get_throttle(brake, target_speed, speed)

    steer = self._get_steer(brake, route_list, pos, input_data['compass'], speed)

    # control = carla.VehicleControl()
    # control.steer = steer
    # control.throttle = throttle
    # control.brake = float(brake or control_brake)

    self.steer = steer
    self.throttle = throttle
    self.brake = brake
    self.target_speed = target_speed
    
    # control_elements_list = []
    # control_elements = {}
    # control_elements['throttle'] = throttle
    # control_elements['steer'] = steer
    # control_elements['brake'] = brake
    # control_elements_list.append(control_elements)
    
    control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    input_data['control'] = control
  
  def _get_steer(self, brake,  route, pos, theta, speed, restore=True):
    
    if len(route) == 1:
      near_target = np.array([route[0][0].transform.location.x, route[0][0].transform.location.y])
    else:
      near_target = np.array([route[1][0].transform.location.x, route[1][0].transform.location.y])
       
    # if self._waypoint_planner.is_last:  # end of route
    #   angle = 0.0
   
    if (speed < 0.01) and brake:  # prevent accumulation
      near_angle = 0.0
    else:
      near_angle_unnorm = self._get_angle_to(pos, theta, near_target)
      near_angle = near_angle_unnorm / 90

    # self.aim_wp = target
    # self.angle = angle

    if restore:
      self._turn_controller.load()
    near_steer = self._turn_controller.step(near_angle)
    if restore:
      self._turn_controller.save()
    

    steer = np.clip(near_steer, -1.0, 1.0)
    return steer

  def _get_steer_extrapolation(self, route, pos, theta, restore=True):
    if self._waypoint_planner_extrapolation.is_last:  # end of route
      angle = 0.0
    else:
      if len(route) == 1:
        target = route[0][0]
      else:
        target = route[1][0]

      angle_unnorm = self._get_angle_to(pos, theta, target)
      angle = angle_unnorm / 90

    if restore:
      self._turn_controller_extrapolation.load()
    steer = self._turn_controller_extrapolation.step(angle)
    if restore:
      self._turn_controller_extrapolation.save()

    steer = np.clip(steer, -1.0, 1.0)
    steer = round(steer, 3)

    return steer

  def _get_throttle(self, brake, target_speed, speed, restore=True):
    control_brake = False
    if (speed / target_speed) > self.config.brake_ratio:
      control_brake = True

    target_speed = target_speed if not brake else 0.0

    # if self._waypoint_planner.is_last:  # end of route
    #   target_speed = 0.0

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    if restore:
      self._speed_controller.load()
    throttle = self._speed_controller.step(delta)
    if restore:
      self._speed_controller.save()

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    if brake:
      throttle = 0.0

    return throttle, control_brake

  def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
    if self._waypoint_planner_extrapolation.is_last:  # end of route
      target_speed = 0.0

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    if restore:
      self._speed_controller_extrapolation.load()
    throttle = self._speed_controller_extrapolation.step(delta)
    if restore:
      self._speed_controller_extrapolation.save()

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    return throttle
    
  def _get_brake(self, near_command=None):
    actors = self._world.get_actors()
    ego_speed = self._get_forward_speed()

    ego_vehicle_location = self.actor.get_location()
    ego_vehicle_transform = self.actor.get_transform()

    center_ego_bb_global = ego_vehicle_transform.transform(self.actor.bounding_box.location)
    ego_bb_global = carla.BoundingBox(center_ego_bb_global, self.actor.bounding_box.extent)
    ego_bb_global.rotation = ego_vehicle_transform.rotation

    if self.visualize == 1:
      self._world.debug.draw_box(box=ego_bb_global,
                                 rotation=ego_bb_global.rotation,
                                 thickness=0.1,
                                 color=carla.Color(0, 0, 0, 255),
                                 life_time=(1.0 / self.config.carla_fps))

    self.stop_sign_close = False
    self.walker_close = False
    self.vehicle_hazard = False
    self.vehicle_ids = []
    self.walker_hazard = False

    # distance in which we check for collisions
    number_of_future_frames = int(self.config.extrapolation_seconds * self.config.bicycle_frame_rate)
    number_of_future_frames_no_junction = int(self.config.extrapolation_seconds_no_junction *
                                              self.config.bicycle_frame_rate)

    # Get future bbs of walkers
    nearby_walkers = self.forcast_walkers(actors, ego_vehicle_location, ego_vehicle_transform,
                                          number_of_future_frames, number_of_future_frames_no_junction)

    # Get future bbs of ego_vehicle
    bounding_boxes_front, bounding_boxes_back, future_steering = self.forcast_ego_agent(
        ego_vehicle_transform, ego_speed, number_of_future_frames, number_of_future_frames_no_junction)

    # -----------------------------------------------------------
    # Vehicle detection
    # -----------------------------------------------------------
    vehicles = actors.filter('*vehicle*')
    nearby_vehicles = {}
    tmp_near_vehicle_id = []
    tmp_stucked_vehicle_id = []
    for vehicle in vehicles:
      if vehicle.id == self.actor.id:
        continue
      if vehicle.get_location().distance(ego_vehicle_location) < self.config.detection_radius:
        tmp_near_vehicle_id.append(vehicle.id)
        veh_future_bbs = []
        traffic_transform = vehicle.get_transform()
        traffic_control = vehicle.get_control()
        traffic_velocity = vehicle.get_velocity()
        traffic_speed = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity)  # In m/s

        self.vehicle_speed_buffer[vehicle.id]['velocity'].append(traffic_speed)
        self.vehicle_speed_buffer[vehicle.id]['throttle'].append(traffic_control.throttle)
        self.vehicle_speed_buffer[vehicle.id]['brake'].append(traffic_control.brake)
        if len(self.vehicle_speed_buffer[vehicle.id]['velocity']) > self.config.stuck_buffer_size:
          self.vehicle_speed_buffer[vehicle.id]['velocity'] = self.vehicle_speed_buffer[
              vehicle.id]['velocity'][-self.config.stuck_buffer_size:]
          self.vehicle_speed_buffer[vehicle.id]['throttle'] = self.vehicle_speed_buffer[
              vehicle.id]['throttle'][-self.config.stuck_buffer_size:]
          self.vehicle_speed_buffer[vehicle.id]['brake'] = self.vehicle_speed_buffer[
              vehicle.id]['brake'][-self.config.stuck_buffer_size:]

        # Safety box that models the safety distance that other traffic participants keep.
        if self.config.model_interactions:
          traffic_safety_loc = np.array([self.config.traffic_safety_box_length, 0.0])
          center_safety_box = traffic_transform.transform(
              carla.Location(x=traffic_safety_loc[0], y=traffic_safety_loc[1], z=vehicle.bounding_box.location.z))
          traffic_safety_extent = carla.Vector3D(
              vehicle.bounding_box.extent.x,
              vehicle.bounding_box.extent.y * self.config.traffic_safety_box_width_multiplier,
              vehicle.bounding_box.extent.z)
          traffic_safety_box = carla.BoundingBox(center_safety_box, traffic_safety_extent)
          traffic_safety_box.rotation = traffic_transform.rotation

        action = np.array([traffic_control.steer, traffic_control.throttle, traffic_control.brake])

        traffic_safety_color = carla.Color(0, 255, 0, 255)
        if self.config.model_interactions and self.check_obb_intersection(traffic_safety_box, ego_bb_global):
          traffic_safety_color = carla.Color(255, 0, 0, 255)
          # Set action to break to model interactions
          action[1] = 0.0
          action[2] = 1.0

        if self.visualize == 1 and self.config.model_interactions:
          self._world.debug.draw_box(box=traffic_safety_box,
                                      rotation=traffic_safety_box.rotation,
                                      thickness=0.1,
                                      color=traffic_safety_color,
                                      life_time=(1.0 / self.config.carla_fps))

        next_loc = np.array([traffic_transform.location.x, traffic_transform.location.y])

        next_yaw = np.array([np.deg2rad(traffic_transform.rotation.yaw)])
        next_speed = np.array([traffic_speed])

        for i in range(number_of_future_frames):
          if not self.junction and (i > number_of_future_frames_no_junction):
            break

          next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)

          delta_yaws = np.rad2deg(next_yaw).item()

          transform = carla.Transform(
              carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z),
              carla.Rotation(pitch=traffic_transform.rotation.pitch,
                              yaw=delta_yaws,
                              roll=traffic_transform.rotation.roll))

          # Safety box that models the safety distance that other traffic
          # participants keep.
          if self.config.model_interactions:
            center_safety_box = transform.transform(
                carla.Location(x=traffic_safety_loc[0], y=traffic_safety_loc[1], z=vehicle.bounding_box.location.z))
            traffic_safety_box = carla.BoundingBox(center_safety_box, traffic_safety_extent)
            traffic_safety_box.rotation = transform.rotation

          if self.config.model_interactions and (
              self.check_obb_intersection(traffic_safety_box, bounding_boxes_back[i]) or
              self.check_obb_intersection(traffic_safety_box, bounding_boxes_front[i])):
            traffic_safety_color = carla.Color(255, 0, 0, 255)
            # Set action to break to model interactions
            action[1] = 0.0
            action[2] = 1.0
          else:
            traffic_safety_color = carla.Color(0, 255, 0, 255)
            action[1] = traffic_control.throttle
            action[2] = traffic_control.brake

          if self.visualize == 1 and self.config.model_interactions:
            self._world.debug.draw_box(box=traffic_safety_box,
                                        rotation=traffic_safety_box.rotation,
                                        thickness=0.1,
                                        color=traffic_safety_color,
                                        life_time=(1.0 / self.config.carla_fps))

          bounding_box = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
          bounding_box.rotation = transform.rotation

          color = carla.Color(0, 0, 255, 255)
          if self.visualize == 1:
            self._world.debug.draw_box(box=bounding_box,
                                        rotation=bounding_box.rotation,
                                        thickness=0.1,
                                        color=color,
                                        life_time=(1.0 / self.config.carla_fps))
          veh_future_bbs.append(bounding_box)

        if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]['velocity']) < self.config.stuck_vel_threshold and
            statistics.mean(self.vehicle_speed_buffer[vehicle.id]['throttle']) > self.config.stuck_throttle_threshold
            and statistics.mean(self.vehicle_speed_buffer[vehicle.id]['brake']) < self.config.stuck_brake_threshold):
          tmp_stucked_vehicle_id.append(vehicle.id)

        nearby_vehicles[vehicle.id] = veh_future_bbs

    # delete old vehicles
    to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
    for d in to_delete:
      del self.vehicle_speed_buffer[d]
    # -----------------------------------------------------------
    # Intersection checks with ego vehicle
    # -----------------------------------------------------------
    back_only_vehicle_id = []

    color = carla.Color(0, 255, 0, 255)
    color2 = carla.Color(0, 255, 255, 255)
    for i, elem in enumerate(zip(bounding_boxes_front, bounding_boxes_back)):
      bounding_box, bounding_box_back = elem
      i_stuck = i
      for vehicle_id, traffic_participant in nearby_vehicles.items():
        if not self.junction and (i > number_of_future_frames_no_junction):
          break
        if vehicle_id in tmp_stucked_vehicle_id:
          i_stuck = 0
        back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) is True)
        front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]))
        # During lane changes we consider collisions with the back side
        if back_intersect:
          color2 = carla.Color(255, 0, 0, 0)
          if self.junction or (i <= number_of_future_frames_no_junction):
            self.vehicle_hazard = True
            if vehicle_id not in self.vehicle_ids:
              self.vehicle_ids.append(vehicle_id)
        if vehicle_id in back_only_vehicle_id:
          back_only_vehicle_id.remove(vehicle_id)
          if back_intersect:
            back_only_vehicle_id.append(vehicle_id)
          continue
        if back_intersect and not front_intersect:
          back_only_vehicle_id.append(vehicle_id)
        if front_intersect:
          color = carla.Color(255, 0, 0, 255)
          if self.junction or (i <= number_of_future_frames_no_junction):
            self.vehicle_hazard = True
            if vehicle_id not in self.vehicle_ids:
              self.vehicle_ids.append(vehicle_id)

      for walker in nearby_walkers:
        if not self.junction and (i > number_of_future_frames_no_junction):
          break
        if self.check_obb_intersection(bounding_box, walker[i]):
          color = carla.Color(255, 0, 0, 255)
          if self.junction or (i <= number_of_future_frames_no_junction):
            self.walker_hazard = True

      if self.visualize == 1:
        self._world.debug.draw_box(box=bounding_box,
                                    rotation=bounding_box.rotation,
                                    thickness=0.1,
                                    color=color,
                                    life_time=(1.0 / self.config.carla_fps))
        self._world.debug.draw_box(box=bounding_box_back,
                                    rotation=bounding_box.rotation,
                                    thickness=0.1,
                                    color=color2,
                                    life_time=(1.0 / self.config.carla_fps))

    # -----------------------------------------------------------
    # Safety box
    # -----------------------------------------------------------

    # add safety bounding box in front.
    # If there is anything in there we won't start driving
    color = carla.Color(0, 255, 0, 255)

    # Bremsweg formula for emergency break
    bremsweg = (((ego_speed * 3.6) / 10.0)**2 / 2.0) \
               + self.config.safety_box_safety_margin

    index_future_orientation = int((bremsweg / self.target_speed) \
                                 * self.config.bicycle_frame_rate)

    index_drive_margin = int((1.0 / self.target_speed) * self.config.bicycle_frame_rate)
    next_loc_brake = np.array([0.0, 0.0])
    next_yaw_brake = np.array(0.0)
    next_speed_brake = np.array(self.target_speed)
    action_brake = np.array(np.stack([self.steer, 0.0, 0.0], axis=-1))

    for o in range(min(index_drive_margin + index_future_orientation, number_of_future_frames)):
      if o == index_drive_margin:
        action_brake[2] = 1.0

      next_loc_brake, next_yaw_brake, next_speed_brake = self.ego_model.forward(next_loc_brake, next_yaw_brake,
                                                                                next_speed_brake, action_brake)
      index = o
      if o >= len(future_steering):
        index = len(future_steering) - 1
      action_brake[0] = future_steering[index]

    center_safety_box = ego_vehicle_transform.transform(
        carla.Location(x=next_loc_brake[0], y=next_loc_brake[1], z=self.actor.bounding_box.location.z))
    bounding_box = carla.BoundingBox(center_safety_box, self.actor.bounding_box.extent)
    bounding_box.rotation = ego_vehicle_transform.rotation
    bounding_box.rotation.yaw = float(
        t_u.normalize_angle_degree(np.rad2deg(next_yaw_brake) + bounding_box.rotation.yaw))

    for participant_id, traffic_participant in nearby_vehicles.items():
      # check the first BB of the traffic participant.
      if self.check_obb_intersection(bounding_box, traffic_participant[0]):
        color = carla.Color(255, 0, 0, 255)
        self.vehicle_hazard = True
        if participant_id not in self.vehicle_ids:
          self.vehicle_ids.append(participant_id)

    for walker in nearby_walkers:
      # check the first BB of the traffic participant.
      if self.check_obb_intersection(bounding_box, walker[0]):
        color = carla.Color(255, 0, 0, 255)
        self.walker_hazard = True
        if walker.id not in self.vehicle_ids:
          self.vehicle_ids.append(walker.id)

    # -----------------------------------------------------------
    # Red light detection
    # -----------------------------------------------------------
    # The safety box is also used for red light detection
    # self.traffic_light_hazard = self.ego_agent_affected_by_red_light(ego_vehicle_transform, bounding_box)
    if self.traffic_light_hazard:
      color = carla.Color(255, 0, 0, 255)

    # -----------------------------------------------------------
    # Stop sign detection
    # -----------------------------------------------------------
    # self.stop_sign_hazard = self.ego_agent_affected_by_stop_sign(ego_vehicle_transform, ego_vehicle_location, actors,
    #                                                              ego_speed, bounding_box)

    if self.visualize == 1:
      self._world.debug.draw_box(box=bounding_box,
                                 rotation=bounding_box.rotation,
                                 thickness=0.1,
                                 color=color,
                                 life_time=(1.0 / self.config.carla_fps))

    return self.vehicle_hazard or self.traffic_light_hazard or self.walker_hazard or self.stop_sign_hazard

  def forcast_ego_agent(self, vehicle_transform, speed, number_of_future_frames, number_of_future_frames_no_junction):
    next_loc_no_brake = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
    next_yaw_no_brake = np.array([np.deg2rad(vehicle_transform.rotation.yaw)])
    next_speed_no_brake = np.array([speed])

    # NOTE intentionally set ego vehicle to move at the target speed
    # (we want to know if there is an intersection if we would not brake)
    throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)

    action_no_brake = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

    future_steering = []
    bounding_boxes_front = []
    bounding_boxes_back = []
    for i in range(number_of_future_frames):
      if not self.junction and (i > number_of_future_frames_no_junction):
        break

      # calculate ego vehicle bounding box for the next timestep.
      # We don't consider timestep 0 because it is from the past and has already happened.
      next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(
          next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)

      waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(next_loc_no_brake)
      steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp,
                                                               next_loc_no_brake,
                                                               next_yaw_no_brake.item(),
                                                               restore=False)
      throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed,
                                                                     next_speed_no_brake,
                                                                     restore=False)
      brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
      action_no_brake = np.array(
          np.stack([steer_extrapolation_temp,
                    float(throttle_extrapolation_temp), brake_extrapolation_temp], axis=-1))
      if brake_extrapolation_temp:
        future_steering.append(0.0)
      else:
        future_steering.append(steer_extrapolation_temp)

      delta_yaws_no_brake = np.rad2deg(next_yaw_no_brake).item()
      cosine = np.cos(next_yaw_no_brake.item())
      sine = np.sin(next_yaw_no_brake.item())

      extent = self.actor.bounding_box.extent
      extent.x = extent.x / 2.

      # front half
      transform = carla.Transform(
          carla.Location(x=next_loc_no_brake[0].item() + extent.x * cosine,
                         y=next_loc_no_brake[1].item() + extent.y * sine,
                         z=vehicle_transform.location.z))
      bounding_box = carla.BoundingBox(transform.location, extent)
      bounding_box.rotation = carla.Rotation(pitch=vehicle_transform.rotation.pitch,
                                             yaw=delta_yaws_no_brake,
                                             roll=vehicle_transform.rotation.roll)

      # back half
      transform_back = carla.Transform(
          carla.Location(x=next_loc_no_brake[0].item() - extent.x * cosine,
                         y=next_loc_no_brake[1].item() - extent.y * sine,
                         z=vehicle_transform.location.z))
      bounding_box_back = carla.BoundingBox(transform_back.location, extent)
      bounding_box_back.rotation = carla.Rotation(pitch=vehicle_transform.rotation.pitch,
                                                  yaw=delta_yaws_no_brake,
                                                  roll=vehicle_transform.rotation.roll)

      bounding_boxes_front.append(bounding_box)
      bounding_boxes_back.append(bounding_box_back)

    return bounding_boxes_front, bounding_boxes_back, future_steering

  def forcast_walkers(self, actors, vehicle_location, vehicle_transform, number_of_future_frames,
                      number_of_future_frames_no_junction):
    walkers = actors.filter('*walker*')
    nearby_walkers = []
    for walker in walkers:
      if walker.get_location().distance(vehicle_location) < self.config.detection_radius:
        # Walkers need 1 frame to be visible, so we wait for 1 frame before we react.
        if not walker.id in self.visible_walker_ids:
          self.visible_walker_ids.append(walker.id)
          continue

        walker_future_bbs = []
        walker_transform = walker.get_transform()

        relative_pos = t_u.get_relative_transform(np.array(vehicle_transform.get_matrix()),
                                                  np.array(walker_transform.get_matrix()))

        # If the walker is in front of us, we want to drive slower, but not if it is behind us.
        if relative_pos[0] > self.config.ego_extent_x:
          self.walker_close = True

        walker_velocity = walker.get_velocity()
        walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
        walker_location = walker_transform.location
        walker_direction = walker.get_control().direction

        if walker.id in self.walker_past_pos:
          real_distance = walker_location.distance(self.walker_past_pos[walker.id])
          if real_distance < 0.0001:
            walker_speed = 0.0  # Walker is stuck somewhere.

        self.walker_past_pos.update({walker.id: walker_location})

        for i in range(number_of_future_frames):
          if not self.junction and i > number_of_future_frames_no_junction:
            break

          new_x = walker_location.x + (walker_direction.x * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          new_y = walker_location.y + (walker_direction.y * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          new_z = walker_location.z + (walker_direction.z * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          walker_location = carla.Location(new_x, new_y, new_z)

          transform = carla.Transform(walker_location)
          bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
          bounding_box.rotation = carla.Rotation(
              pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
              yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw,
              roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)

          color = carla.Color(0, 0, 255, 255)
          if self.visualize == 1:
            self._world.debug.draw_box(box=bounding_box,
                                       rotation=bounding_box.rotation,
                                       thickness=0.1,
                                       color=color,
                                       life_time=(1.0 / self.config.carla_fps))
          walker_future_bbs.append(bounding_box)
        nearby_walkers.append(walker_future_bbs)

    return nearby_walkers

  def ego_agent_affected_by_red_light(self, vehicle_transform, detection_box):
    """
    Checks whether the autopilot is affected by a traffic light and should stop.
    :param vehicle_transform: carla transform object of the ego vehicle
    :param detection_box: carla bounding box used to detect the traffic light.
    :return: True if the agent should stop for a traffic light, False else.
    """
    light_hazard = False
    self._active_traffic_light = None

    vehicle_location = vehicle_transform.location
    self.close_traffic_lights.clear()
    for light, center, waypoints in self.list_traffic_lights:

      center_loc = carla.Location(center)
      if center_loc.distance(vehicle_location) > self.config.light_radius:
        continue

      for wp in waypoints:
        # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
        length_bounding_box = carla.Vector3D((wp.lane_width / 2.0) * 0.9, light.trigger_volume.extent.y,
                                             light.trigger_volume.extent.z)

        bounding_box = carla.BoundingBox(wp.transform.location, length_bounding_box)

        gloabl_rot = light.get_transform().rotation
        bounding_box.rotation = carla.Rotation(pitch=light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                               yaw=light.trigger_volume.rotation.yaw + gloabl_rot.yaw,
                                               roll=light.trigger_volume.rotation.roll + gloabl_rot.roll)

        center_vehicle = vehicle_transform.transform(self.actor.bounding_box.location)
        vehicle_bb = carla.BoundingBox(center_vehicle, self.actor.bounding_box.extent)
        vehicle_bb.rotation = vehicle_transform.rotation

        affects_ego = False
        if self.check_obb_intersection(detection_box, bounding_box) \
            or self.check_obb_intersection(vehicle_bb, bounding_box):
          affects_ego = True
          if light.state in (carla.libcarla.TrafficLightState.Red, carla.libcarla.TrafficLightState.Yellow):
            self._active_traffic_light = light
            light_hazard = True

        self.close_traffic_lights.append([bounding_box, light.state, light.id, affects_ego])

        if self.visualize == 1:
          if light.state == carla.libcarla.TrafficLightState.Red:
            color = carla.Color(255, 0, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Yellow:
            color = carla.Color(255, 255, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Green:
            color = carla.Color(0, 255, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Off:
            color = carla.Color(0, 0, 0, 255)
          else:  # unknown
            color = carla.Color(0, 0, 255, 255)

          self._world.debug.draw_box(box=bounding_box,
                                     rotation=bounding_box.rotation,
                                     thickness=0.1,
                                     color=color,
                                     life_time=(1.0 / self.config.carla_fps))

          self._world.debug.draw_point(wp.transform.location + carla.Location(z=light.trigger_volume.location.z),
                                       size=0.1,
                                       color=color,
                                       life_time=0.01)

    return light_hazard

  def ego_agent_affected_by_stop_sign(self, vehicle_transform, vehicle_location, actors, speed, safety_box):
    stop_sign_hazard = False
    self.close_stop_signs.clear()
    stop_signs = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.config.light_radius)
    center_vehicle_stop_sign_detector_bb = vehicle_transform.transform(self.actor.bounding_box.location)
    extent_vehicle_stop_sign_detector_bb = self.actor.bounding_box.extent
    vehicle_stop_sign_detector_bb = carla.BoundingBox(center_vehicle_stop_sign_detector_bb,
                                                      extent_vehicle_stop_sign_detector_bb)
    vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation

    for stop_sign in stop_signs:
      center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
      transform_stop_sign = carla.Transform(center_bb_stop_sign)
      bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, stop_sign.trigger_volume.extent)
      rotation_stop_sign = stop_sign.get_transform().rotation
      bounding_box_stop_sign.rotation = carla.Rotation(
          pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch,
          yaw=stop_sign.trigger_volume.rotation.yaw + rotation_stop_sign.yaw,
          roll=stop_sign.trigger_volume.rotation.roll + rotation_stop_sign.roll)

      color = carla.Color(0, 255, 0, 255)

      affects_ego = False
      if self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign):
        if not stop_sign.id in self.cleared_stop_signs:
          affects_ego = True
          self.stop_sign_close = True
          if (speed * 3.6) > 0.0:  # Conversion from m/s to km/h
            stop_sign_hazard = True
            color = carla.Color(255, 0, 0, 255)
          else:
            self.cleared_stop_signs.append(stop_sign.id)
      elif self.check_obb_intersection(safety_box, bounding_box_stop_sign):
        if not stop_sign.id in self.cleared_stop_signs:
          affects_ego = True
          self.stop_sign_close = True
          color = carla.Color(255, 0, 0, 255)

      self.close_stop_signs.append([bounding_box_stop_sign, stop_sign.id, affects_ego])

      if self.visualize:
        self._world.debug.draw_box(box=bounding_box_stop_sign,
                                   rotation=bounding_box_stop_sign.rotation,
                                   thickness=0.1,
                                   color=color,
                                   life_time=(1.0 / self.config.carla_fps))

    # reset past cleared stop signs
    for cleared_stop_sign in self.cleared_stop_signs:
      remove_stop_sign = True
      for stop_sign in stop_signs:
        if stop_sign.id == cleared_stop_sign:
          # stop sign is still around us hence it might be active
          remove_stop_sign = False
      if remove_stop_sign:
        self.cleared_stop_signs.remove(cleared_stop_sign)

    return stop_sign_hazard

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

  def dot_product(self, vector1, vector2):
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

  def cross_product(self, vector1, vector2):
    return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y,
                          y=vector1.z * vector2.x - vector1.x * vector2.z,
                          z=vector1.x * vector2.y - vector1.y * vector2.x)

  def get_separating_plane(self, r_pos, plane, obb1, obb2):
    """ Checks if there is a separating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        """
    return (abs(self.dot_product(r_pos, plane)) >
            (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
             abs(self.dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane)) +
             abs(self.dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane)) +
             abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
             abs(self.dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane)) +
             abs(self.dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane))))

  def check_obb_intersection(self, obb1, obb2):
    """General algorithm that checks if 2 3D oriented bounding boxes intersect."""
    r_pos = obb2.location - obb1.location
    return not (
        self.get_separating_plane(r_pos, obb1.rotation.get_forward_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb1.rotation.get_right_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb1.rotation.get_up_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_forward_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_right_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_up_vector(), obb1, obb2) or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1,
            obb2) or self.get_separating_plane(
                r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1,
                obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1, obb2)
        or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2)
        or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
        self.get_separating_plane(r_pos, self.cross_product(obb1.rotation.get_up_vector(),
                                                            obb2.rotation.get_up_vector()), obb1, obb2))

  def _get_angle_to(self, pos, theta, target):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    diff = target - pos
    aim_0 = (cos_theta * diff[0] + sin_theta * diff[1])
    aim_1 = (-sin_theta * diff[0] + cos_theta * diff[1])

    angle = -math.degrees(math.atan2(-aim_1, aim_0))
    angle = np.float_(angle)
    return angle

  def get_nearby_object(self, vehicle_position, actor_list, radius):
    nearby_objects = []
    for actor in actor_list:
      trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
      trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                              y=trigger_box_global_pos.y,
                                              z=trigger_box_global_pos.z)
      if trigger_box_global_pos.distance(vehicle_position) < radius:
        nearby_objects.append(actor)
    return nearby_objects

class EgoModel():
  """
    Kinematic bicycle model describing the motion of a car given it's state and
    action. Tuned parameters are taken from World on Rails.
    """

  def __init__(self, dt=1. / 4):
    self.dt = dt

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275

    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    if brake:
      accel = self.brake_accel
    else:
      accel = self.throt_accel * throt

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
    next_spds = spds + accel * self.dt
    next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

    next_locs = np.array([next_locs_0, next_locs_1])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds
  
