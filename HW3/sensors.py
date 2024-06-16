
import carla
import numpy as np
import math
import weakref
from carla import ColorConverter as cc
from copy import deepcopy

assigned_location_dict = {'E1': (13.700, 2.600),
						'E2': (13.700, 6.000),
						'E3': (47.200, 1.800),
						'E4': (47.200, 5.100),
						'A1': (31.600, 18.100),
						'A2': (35.100, 18.100),
						'A3': (31.400, -18.100),
						'A4': (34.900, -18.100),
						'B1': (27.900, -18.100),
						'B2': (24.400, -18.100),
						'B3': (28.200, 18.100),
						'B4': (24.600, 18.100),
						'C1': (47.200, -1.700),
						'C2': (47.200, -5.200),
						'C3': (10.700, -0.900),
						'C4': (10.700, -4.400),
						'center': (29.550, 0.85)
						}

def get_actor_display_name(actor, truncate=250):
	name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
	return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- CameraSensor -----------------------------------------------------------------
# ==============================================================================
class SensorManager(object):
	def __init__(self, parent_actor, sensor_spec_list):
		self._parent = parent_actor
		self.id = parent_actor.id
		world = parent_actor.get_world()

		bp_library = world.get_blueprint_library()
		Attachment = carla.AttachmentType

		self.sensors_dict = {}
		self.data = {}
		self.use_lidar = False

		for sensor_spec in sensor_spec_list:
			sensor_transform = (carla.Transform(carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z']), carla.Rotation(roll=sensor_spec['roll'], pitch=sensor_spec['pitch'], yaw=sensor_spec['yaw'])), Attachment.Rigid)	
			sensor_id = sensor_spec['id']
			if sensor_spec['type'].startswith('sensor.camera'):
				sensor_rgb_bp = bp_library.find(str(sensor_spec['type']))
				sensor_rgb_bp.set_attribute('image_size_x', str(sensor_spec['width']))
				sensor_rgb_bp.set_attribute('image_size_y', str(sensor_spec['height']))
				sensor_rgb_bp.set_attribute('fov', str(sensor_spec['fov']))
				sensor_rgb_bp.set_attribute('lens_circle_multiplier', str(3.0))
				sensor_rgb_bp.set_attribute('lens_circle_falloff', str(3.0))
				sensor_rgb_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
				sensor_rgb_bp.set_attribute('chromatic_aberration_offset', str(0))
				if sensor_id.startswith("rgb_center"):
					self.sensor_instance_rgb = world.spawn_actor(
					sensor_rgb_bp,
					sensor_transform[0],
				)
				else:
					self.sensor_instance_rgb = world.spawn_actor(
					sensor_rgb_bp,
					sensor_transform[0],
					attach_to=self._parent,
					attachment_type=sensor_transform[1]
				)
				self.data['image'] = None
				# print("create rgb sensor")
				# self.sensors_dict[sensor_id] = sensor_instance_rgb
    
			if sensor_spec['type'].startswith('sensor.lidar'):
				self.use_lidar = True
				sensor_lidar_bp = bp_library.find(str(sensor_spec['type']))
				sensor_lidar_bp.set_attribute('range', str(85))
				sensor_lidar_bp.set_attribute('rotation_frequency', str(10))
				sensor_lidar_bp.set_attribute('channels', str(64))
				sensor_lidar_bp.set_attribute('upper_fov', str(10))
				sensor_lidar_bp.set_attribute('lower_fov', str(-30))
				sensor_lidar_bp.set_attribute('points_per_second', str(600000))
				sensor_lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
				sensor_lidar_bp.set_attribute('dropoff_general_rate', str(0.45))
				sensor_lidar_bp.set_attribute('dropoff_intensity_limit', str(0.8))
				sensor_lidar_bp.set_attribute('dropoff_zero_intensity', str(0.4))

				self.sensor_instance_lidar = world.spawn_actor(
					sensor_lidar_bp,
					sensor_transform[0],
					attach_to=self._parent,
					attachment_type=sensor_transform[1]
				)
				self.data['lidar'] = None
			
    
	  # We need to pass the lambda a weak reference to self to avoid
		# circular reference.
		weak_self = weakref.ref(self)

		self.sensor_instance_rgb.listen(lambda data: SensorManager._parse_data(weak_self, data, "image"))
	
		if self.use_lidar:
			self.sensor_instance_lidar.listen(lambda data: SensorManager._parse_data(weak_self, data, 'lidar'))
		# print(self.sensors_dict)
		# for sensor_id, sensor in self.sensors_dict.items():
		# 		sensor.listen(lambda data: SensorManager._parse_data(weak_self, data, sensor_id))
  
  
	def destroy(self):
		self.sensor_instance_rgb.destroy()
		if self.use_lidar:
			self.sensor_instance_lidar.destroy()
		del self.data
  
	def get_data(self, frame, sensor_id=None):
		while True:
			if not self.data[sensor_id]:
				# print(f'wait for {self.id} {sensor_id} sensor at frame {frame}')
				continue
			if self.data[sensor_id].frame == frame:
				break

		if sensor_id == "image":
			self.data[sensor_id].convert(cc.Raw)
			array = np.frombuffer(self.data[sensor_id].raw_data, dtype=np.dtype("uint8"))
			array = deepcopy(array)
			array = np.reshape(array, (self.data[sensor_id].height, self.data[sensor_id].width, 4))		# array = array[:, :, ::-1]
			return array
		else:
			points = np.frombuffer(self.data[sensor_id].raw_data, dtype=np.dtype('f4'))
			points = deepcopy(points)
			points = np.reshape(points, (int(points.shape[0] / 4), 4))
			return points			

	@staticmethod
	def _parse_data(weak_self, data, sensor_id):
		self = weak_self()
		if not self:
			return
		
		if sensor_id == 'image':
			self.data['image'] = data	
		if sensor_id == 'lidar':
			self.data['lidar'] = data


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
	def __init__(self, parent_actor):
		self.sensor = None
		self.frame_history = []
		self.id_history = []
		self._parent = parent_actor
		# self.hud = hud
		self.other_actor_id = 0  # init as 0 for static object
		self.other_actor_ids = []  # init as 0 for static object
		self.wrong_collision = False

		self.true_collision = False
		world = self._parent.get_world()
		bp = world.get_blueprint_library().find('sensor.other.collision')
		self.sensor = world.spawn_actor(
				bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(
				lambda event: CollisionSensor._on_collision(weak_self, event))
		self.collision = False

		self.collision_actor_id = None
		self.collision_actor_type = None

	@staticmethod
	def _on_collision(weak_self, event):
		self = weak_self()
		if not self:
			return
		actor_type = get_actor_display_name(event.other_actor)
		# self.hud.notification('Collision with %r' % actor_type)
		# impulse = event.normal_impulse
		# intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
		# dict: {data1, data2}
		# data = frame: {timestamp, other_actor's id, intensity}
		if event.frame not in self.frame_history:
			self.frame_history.append(event.frame)
			self.id_history.append([event.other_actor.id])
		else:
			if event.other_actor.id not in self.id_history[-1]:
				self.id_history[-1].append(event.other_actor.id)
   
		# if len(self.history) > 4000:
		#     self.history.pop(0)
		self.collision = True
		self.collision_actor_id = event.other_actor.id
		self.collision_actor_type = actor_type

		if event.other_actor.id in self.other_actor_ids:
			self.true_collision = True
		if event.other_actor.id == self.other_actor_id: 
			self.true_collision = True
		if event.other_actor.id != self.other_actor_id:
			self.wrong_collision = True


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
	def __init__(self, parent_actor):
		self.sensor = None
		self._parent = parent_actor
		self.lat = 0.0
		self.lon = 0.0
		world = self._parent.get_world()
		bp = world.get_blueprint_library().find('sensor.other.gnss')
		self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)))
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

	@staticmethod
	def _on_gnss_event(weak_self, event):
		self = weak_self()
		if not self:
			return
		self.lat = event.latitude
		self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================
class IMUSensor(object):
	def __init__(self, parent_actor):
		self.sensor = None
		self._parent = parent_actor
		self.accelerometer = (0.0, 0.0, 0.0)
		self.gyroscope = (0.0, 0.0, 0.0)
		self.compass = 0.0
		world = self._parent.get_world()
		bp = world.get_blueprint_library().find('sensor.other.imu')
		self.sensor = world.spawn_actor(
			bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(
			lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

	@staticmethod
	def _IMU_callback(weak_self, sensor_data):
		self = weak_self()
		if not self:
			return
		limits = (-99.9, 99.9)
		self.accelerometer = (
			max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
			max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
			max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
		self.gyroscope = (
			max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
			max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
			max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
		self.compass = math.degrees(sensor_data.compass)
