"""
Some utility functions e.g. for normalizing angles
Functions for detecting red lights are adapted from scenario runners
atomic_criteria.py
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from shapely.geometry import Polygon
import shapely
from PIL import Image
from pathlib import Path
import transforms3d

def normalize_angle(x):
	x = x % (2 * np.pi)  # force in range [0, 2 pi)
	if x > np.pi:  # move to [-pi, pi)
		x -= 2 * np.pi
	return x


def get_relative_pos(ego_matrix, vehicle_matrix):
	"""
	Returns the position of the vehicle matrix in the ego coordinate system.
	:param ego_matrix: ndarray 4x4 Matrix of the ego vehicle in global
	coordinates
	:param vehicle_matrix: ndarray 4x4 Matrix of another actor in global
	coordinates
	:return: ndarray position of the other vehicle in the ego coordinate system
	"""
	# TODO 
	'''
	Extracts the translation vectors from both matrices and subtracts them element-wise. 
	This gives the relative position of the other vehicle in global coordinates.
	'''
	relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3] 
	
	# This is done to get the inverse rotation, effectively transforming from the global coordinate system to the ego coordinate system.
	rot = ego_matrix[:3, :3].T
	
	# It rotates the relative position from the global coordinate system into the ego coordinate system.
	relative_pos = rot @ relative_pos
	return relative_pos
	
	# raise NotImplementedError('get_relative_pos not implemented.')


def extract_yaw_from_matrix(matrix):
	"""
 Extracts the yaw from a CARLA world transformation matrix
 In driving scene, most of the cars are on the same x-y plane, therefore, both pitch and role are 0.
 P.S. remember to normalize the angle
 """

	# TODO 
	
	# Extract submatrix represents the rotation component of the transformation.
	rotation_matrix = np.array(matrix)[:3, :3]
	
    # Extract yaw using the rotation matrix
	yaw = normalize_angle(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

	# raise NotImplementedError('extract_yaw_from_matrix not implemented.')

	return yaw


# Taken from https://stackoverflow.com/a/47381058/9173068
def trapez(y, y0, w):
	return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
	# The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
	# If either of these cases are violated, do some switches.
	if abs(c1 - c0) < abs(r1 - r0):
		# Switch x and y, and switch again when returning.
		xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order
		return (yy, xx, val)

	# At this point we know that the distance in columns (x) is greater
	# than that in rows (y). Possibly one more switch if c0 > c1.
	if c0 > c1:
		return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order

	# The following is now always < 1 in abs
	if (c1 - c0) != 0.0:
		slope = (r1 - r0) / (c1 - c0)
	else:
		slope = 0.0

	# Adjust weight by the slope
	w *= np.sqrt(1 + np.abs(slope)) / 2

	# We write y as a function of x, because the slope is always <= 1
	# (in absolute value)
	x = np.arange(c0, c1 + 1, dtype=float)
	if (c1 - c0) != 0.0:
		y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)
	else:
		y = np.zeros_like(x)

	# Now instead of 2 values for y, we have 2*np.ceil(w/2).
	# All values are 1 except the upmost and bottommost.
	thickness = np.ceil(w / 2)
	yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
	xx = np.repeat(x, yy.shape[1])
	vals = trapez(yy, y.reshape(-1, 1), w).flatten()

	yy = yy.flatten()

	# Exclude useless parts and those outside of the interval
	# to avoid parts outside of the picture
	mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

	return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def draw_line(img, start_row, start_column, end_row, end_column, color=(255, 255, 255), thickness=1, rmax=256):

	if start_row == end_row and start_column == end_column:
		rr, cc, val = start_row, start_column, 1.0
	else:
		rr, cc, val = weighted_line(r0=start_row, c0=start_column, r1=end_row, c1=end_column, w=thickness, rmax=rmax)

	img[rr, cc, 0] = val * color[0] + (1.0 - val) * img[rr, cc, 0]
	img[rr, cc, 1] = val * color[1] + (1.0 - val) * img[rr, cc, 1]
	img[rr, cc, 2] = val * color[2] + (1.0 - val) * img[rr, cc, 2]
	return img


def get_corner(box):
	"""
	Get the coordinate of 
 	[top_left, top_right, bottom_right, bottom_left] from 
  x, y, extent_x, extenx_y, and yaw.
  
  return shape: [4, 2]
 	"""
	# TODO
	translation = np.array([[box[0], box[1]]]) # Get the coordinate of the bbox
	width = box[2] 
	height = box[3]
	yaw = box[4]

	rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) # Compute the rotation matrix
	corners = np.array([[-width, -height], [width, -height], [width, height], [-width, height]]) # Get the corners
	corner_global = (rotation_matrix @ corners.T).T + translation # Get the corner in the global coordinate system
	corner_global = corner_global.astype(np.int64)

	return corner_global

	# raise NotImplementedError('get_corner not implemented')

def draw_box(img, box, color=(255, 255, 255), pixel_per_meter=4, thickness=1):

	corner_global = get_corner(box)
	corner_global = corner_global.astype(np.int64)

  # Only the center is guaranteed to be within the image. Need to clip the corner points.
	max_row = img.shape[0]
	max_column = img.shape[1]
	corner_global[:, 0] = np.clip(corner_global[:, 0], a_min=0, a_max=max_row - 1)
	corner_global[:, 1] = np.clip(corner_global[:, 1], a_min=0, a_max=max_column - 1)
	
	img = draw_line(img,
                  start_row=corner_global[0, 0],
                  start_column=corner_global[0, 1],
                  end_row=corner_global[1, 0],
                  end_column=corner_global[1, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
	img = draw_line(img,
                  start_row=corner_global[1, 0],
                  start_column=corner_global[1, 1],
                  end_row=corner_global[2, 0],
                  end_column=corner_global[2, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
	img = draw_line(img,
                  start_row=corner_global[2, 0],
                  start_column=corner_global[2, 1],
                  end_row=corner_global[3, 0],
                  end_column=corner_global[3, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
	img = draw_line(img,
                  start_row=corner_global[3, 0],
                  start_column=corner_global[3, 1],
                  end_row=corner_global[0, 0],
                  end_column=corner_global[0, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
	
	return img

def vehicle_coor_to_img(box, pixels_per_meter, min_x, min_y):
	"""
	Changed a bounding box from the vehicle x front, y right coordinate system
	to the x back, y right coordinate system of an image
	"""
	# TODO
	# Multiply position and extent by pixels_per_meter to convert the unit from meters to pixels
	box[:4] = box[:4] * pixels_per_meter
	
	
	# Compute pixel location that represents 0/0 in the image
	translation = np.array([-(min_x * pixels_per_meter), -(min_y * pixels_per_meter)])
	
	# Shift the coordinates so that the ego_vehicle is at the center of the image
	box[0] = -box[0]
	box[:2] = box[:2] + translation
	box[4] = -box[4]

def rect_polygon(x, y, width, height, angle):
  p = Polygon([(-width, -height), (width, -height), (width, height), (-width, height)])
  return shapely.affinity.translate(shapely.affinity.rotate(p, angle, use_radians=True), x, y)

def iou_bbs(bb1, bb2):
  '''
	Just do the IoU calculation
  '''
  # TODO
  a = rect_polygon(bb1[0], bb1[1], bb1[2], bb1[3], bb1[4])
  b = rect_polygon(bb2[0], bb2[1], bb2[2], bb2[3], bb2[4])
  intersection_area = a.intersection(b).area
  union_area = a.union(b).area
  iou = intersection_area / union_area
  return iou

# raise NotImplementedError("iou_bbs not implemented.")

def quant_to_box(config, pred_bounding_boxes):
	"""Convert a learn auxiliary class to an x,y location of a box"""
	pred_bb_x = F.softmax(pred_bounding_boxes[0][0], dim=1)
	pred_bb_y = F.softmax(pred_bounding_boxes[1][0], dim=1)
	pred_bb_ext_x = F.softmax(pred_bounding_boxes[2][0], dim=1)
	pred_bb_ext_y = F.softmax(pred_bounding_boxes[3][0], dim=1)
	pred_bb_yaw = F.softmax(pred_bounding_boxes[4][0], dim=1)
	pred_bb_speed = F.softmax(pred_bounding_boxes[5][0], dim=1)

	pred_bb_x = torch.argmax(pred_bb_x, dim=1)
	pred_bb_y = torch.argmax(pred_bb_y, dim=1)
	pred_bb_ext_x = torch.argmax(pred_bb_ext_x, dim=1)
	pred_bb_ext_y = torch.argmax(pred_bb_ext_y, dim=1)
	pred_bb_yaw = torch.argmax(pred_bb_yaw, dim=1)
	pred_bb_speed = torch.argmax(pred_bb_speed, dim=1)

	x_step = (config.max_x - config.min_x) / pow(2, config.model_precision_pos)
	y_step = (config.max_y - config.min_y) / pow(2, config.model_precision_pos)
	extent_step = (30) / pow(2, config.model_precision_pos)
	yaw_step = (2 * np.pi) / pow(2, config.model_precision_angle)
	speed_step = (config.max_speed_pred / 3.6) / pow(2, config.model_precision_speed)

	pred_bb_x = pred_bb_x * x_step - config.max_x
	pred_bb_y = pred_bb_y * y_step - config.max_y
	pred_bb_ext_x = pred_bb_ext_x * extent_step
	pred_bb_ext_y = pred_bb_ext_y * extent_step
	pred_bb_yaw = pred_bb_yaw * yaw_step - np.pi
	pred_bb_speed = pred_bb_speed * speed_step
	pred_bb_center = torch.stack((pred_bb_x, pred_bb_y, pred_bb_ext_x, pred_bb_ext_y, pred_bb_yaw, pred_bb_speed), dim=1)

	return pred_bb_center

		
class ConstantModel():
	def __init__(self, dt=0.5):
		self.dt = dt

	def forward(self, bounding_boxes):
		"""													1  30	8
		Input: bounding_boxes of current frame with shape: [b, n, num_attribute]
		Return: future_boxes of the next time step based on constant velocity hypothesis
  	"""
		# TODO
		"""
			Use the formula to predict the future position of cars
		"""
		x_current_positions = bounding_boxes[:, :, 0]
		y_current_positions = bounding_boxes[:, :, 1]
		yaw = bounding_boxes[:,:,4]
		velocities = bounding_boxes[:, :, 5]

		future_boxes = bounding_boxes
		future_boxes[:,:,0] = x_current_positions + velocities * self.dt * torch.cos(yaw)
		future_boxes[:,:,1] = y_current_positions + velocities * self.dt * torch.sin(yaw)

		return future_boxes	
		
		# raise NotImplementedError("ConstantModel.forward not implemented.")

def visualize(
		config,
		args,
		save_path,
		step,
		topdown,
		gt_bbs,
		future_bbs=None):

	metric = {}
	L2_dis = 0.0
	avg_iou = 0.0

	size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
	size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)


	# we scale up the bev image to 512*512
	scale_factor = 2
	origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
	loc_pixels_per_meter = config.pixels_per_meter * scale_factor
	topdown = topdown[0].numpy()
	topdown = cv2.resize(topdown,
											dsize=(topdown.shape[1] * scale_factor, topdown.shape[0] * scale_factor),
											interpolation=cv2.INTER_NEAREST)

	# Draw input boxes
	if gt_bbs is not None:
		gt_bbs = gt_bbs.detach().cpu().numpy()[0]
		real_boxes = gt_bbs.sum(axis=-1) != 0.
		gt_bbs = gt_bbs[real_boxes]

		if future_bbs is not None:
			if args.model == "learn":
				future_bbs = quant_to_box(config, future_bbs)
			else:
				future_bbs = future_bbs[0]
			future_bbs = future_bbs.detach().cpu().numpy()
   
		L2_dis = None
		# TODO 
		"""
			Just calculate the L2 distance, and I notice that 
			the size of gt box and future box may have different size, thus, I discard the exceeding part
		"""
		mLen = min(len(gt_bbs[:, :2]), len(future_bbs[:, :2]))
		
		gt_bbs1 = gt_bbs[:mLen]
		gt_xy = gt_bbs1[:, :2]

		future_bbs1 = future_bbs[:mLen]
		pred_xy = future_bbs1[:, :2]


		L2_dis = np.linalg.norm(gt_xy - pred_xy, axis=1)
		L2_dis = L2_dis[0] if len(L2_dis) != 0 else 0
		if L2_dis is None:
			raise NotImplementedError('calculate L2_dis not implemented')
		# END TODO 
  
		for idx, box in enumerate(gt_bbs):
			if future_bbs is not None:
				future_center = future_bbs[idx]
				avg_iou += iou_bbs(box, future_center)
			
			# TODO 
			# Draw predicted and ground truth bounding boxes on BEV image
			# P.S. use vehicle_coor_to_img(), draw_box()
			vehicle_coor_to_img(box, loc_pixels_per_meter, config.min_x, config.min_y)
			topdown = draw_box(topdown, box, (255, 255, 0))
			vehicle_coor_to_img(future_center, loc_pixels_per_meter, config.min_x, config.min_y)
			topdown = draw_box(topdown, future_center, (0, 0, 255))

		# for box in future_bbs:
		# 	newBox = vehicle_coor_to_img(box, loc_pixels_per_meter, config.min_x, config.min_y)
		# 	topdown = draw_box(topdown, newBox, (0, 0, 255))


		if len(gt_bbs) == 0:
			metric['IOU'] = 0
			metric["L2 distance"] = 0
			valid = False
   
		else:	
			avg_iou /= len(gt_bbs)
			metric['IOU'] = avg_iou
			metric["L2 distance"] = L2_dis
			valid = True

	# Need to sometimes do this so that cv2 doesn't start crying
	topdown = np.ascontiguousarray(topdown, dtype=np.uint8)
	
	topdown = Image.fromarray(topdown.astype(np.uint8))
	store_path = str(str(save_path) + (f'/{step:04}.jpg'))
	Path(store_path).parent.mkdir(parents=True, exist_ok=True)
	if step % 100 == 0:
		topdown.save(store_path)

	return metric, valid