import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd

import numpy as np
import cv2
from collections import deque
import argparse


######### GLOBAL VARIABLES #########
CARTESIAN_GRID_SPACING = 0.5 
REAL_DIAMETER = 0.3
ORIGIN_POINT = [640, 320] # MAYBE IT SHOULD BE THE OPPOSITE
HEMISPHERE_RADIUS = 2.5225
HEMISPHERE_RADIUS = 2.6
DIAMETER_BUFFER = 5 # in pixels

def index_to_ij(index):
	if index <= 4:
		return (0, index + 2)
	if index >= 5 and index <= 11:
		return (1, index - 4)
	if index >= 12 and index <= 20:
		return (2, index - 12)
	if index >= 21 and index <= 29:
		return (3, index - 21)
	if index >= 30 and index <= 38:
		return (4, index - 30)
	if index >= 39 and index <= 47:
		return (5, index - 39)
	if index >= 48 and index <= 56:
		return (6, index - 48)
	if index >= 57 and index <= 63:
		return (7, index - 56)
	if index >= 64 and index <= 68:
		return (8, index - 62) 


def circle_from_three_points(k1, k2, k3):
	A = k1.shifted_z_pos_3D()
	B = k2.shifted_z_pos_3D()
	C = k3.shifted_z_pos_3D()
	a = np.linalg.norm(C-B)
	b = np.linalg.norm(C-A)
	c = np.linalg.norm(B-A)
	s = (a + b + c) / 2
	R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
	b1 = a*a * (b*b + c*c - a*a)
	b2 = b*b * (a*a + c*c - b*b)
	b3 = c*c * (a*a + b*b - c*c)
	P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
	P /= b1 + b2 + b3
	return [R, P]

def xy_distance(k1, k2):
	return np.sqrt((k1.pos_3D()[0] - k2.pos_3D()[0])**2 + (k1.pos_3D()[1] - k2.pos_3D()[1])**2)


def HorNeighIndices(i, j):
	'''
	Returns the indices of the two horizontal neighbour indices
	'''
	if (i, j) in [(0,2), (0,6), (1,1), (1,7), (7,1), (7,7), (8,2), (8,6)] or j == 0 or j == 8:
		return []
	else:
		return [(i, j-1), (i, j+1)]

def VerNeighIndices(i, j):
	'''
	Returns the indices of the two vertical neighbour indices
	'''
	if (i, j) in [(6,0), (2,0), (7,1), (1,1), (1,7), (7,7), (2,8), (6,8)] or i == 0 or i == 8:
		return []
	else:
		return [(i-1, j), (i+1, j)]


class KeypointClass:
	'''
	Class to generalize the keypoints we get from blob detection
	'''
	def __init__(self, keypoint):
		self.point = keypoint
		self.size_history = []
		self.z_offset = 0 
		self.z_multiplier = 1
		self.init_3D = np.array([0, 0, 0])
		self.bbox = (0, 0, 0, 0)
		self.a = 1
		self.b = 0
		self.running_shifted_z = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.hor_dists = [99, 99] # The undisturbed distances to horizontal and vertical dots
		self.ver_dists = [99, 99]
		self.actual_z = 0

	def x(self):
		return self.point.pt[0]

	def y(self):
		return self.point.pt[1]

	def size(self):
		return self.point.size

	def set_params(self, a, b):
		self.a = a
		self.b = b

	def add_size_to_history(self):
		self.size_history.append(self.size())

	def get_size_history(self):
		return self.size_history

	def set_hor_dists(self, h0, h1):
		self.hor_dists = [h0, h1]

	def set_ver_dists(self, v0, v1):
		self.ver_dists = [v0, v1]

	def r(self):
		'''
		THE SIZE MEASURE IS ACTUALLY THE DIAMETER
		'''
		return self.point.size / 2

	def pos_3D(self):
		pixel_area = self.size()
		pixel_diameter = 2 * np.sqrt(pixel_area / np.pi)

		x_offset = self.x() - ORIGIN_POINT[0]
		y_offset = ORIGIN_POINT[1] - self.y()

		# real_z = -4.7507*pixel_diameter + 45.408
		real_z = (-1 / self.a) * self.size() + (self.b / self.a)


		# This value needs calibration too !!!
		# real_x = (REAL_DIAMETER * x_offset / pixel_diameter) / 5.5
		# real_y = (REAL_DIAMETER * y_offset / pixel_diameter) / 5.5
		real_x = (REAL_DIAMETER * x_offset / self.size()) 
		real_y = (REAL_DIAMETER * y_offset / self.size()) 
		return np.array([real_x, real_y, real_z])

	def shifted_z_pos_3D(self):
		p = self.pos_3D()

		# Pure offset
		if not np.isnan(self.z_offset):
			shifted_z = p[2] + self.z_offset
			self.running_shifted_z.pop(0)
			self.running_shifted_z.append(shifted_z)
			p[2] = np.mean(self.running_shifted_z)
			# p[2] = shifted_z

		# Try multiplier instead
		# if not np.isnan(self.z_multiplier):
		# 	p[2] = p[2] * self.z_multiplier

		# Try combination of the two
		# if not np.isnan(self.z_multiplier):
		# 	p[2] = (p[2] - 10) * self.z_multiplier 

		return p


	def update_initial_pos(self):
		self.init_3D = self.pos_3D()
		actual_z = np.sqrt(HEMISPHERE_RADIUS**2 - self.init_3D[0]**2 - self.init_3D[1]**2)
		self.actual_z = actual_z
		# print self.size(), actual_z

		self.z_offset = actual_z - self.init_3D[2]

		average_z_offset = - 10 # Calculated from the average
		z_diff = self.init_3D[2] + average_z_offset

		self.z_multiplier = actual_z / self.init_3D[2]
		# self.z_multiplier = actual_z / z_diff

		# self.z_offset = 0
		# self.z_multiplier = 1


	def update_bbox(self, bbox):
		self.bbox = bbox


######### CAMERA SETUP #########
cap = cv2.VideoCapture(0)

######### BLOB DETECTION SETUP #########
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.maxArea = 5000
params.minArea = 50
params.filterByCircularity = True
params.minCircularity = 0.8;
detector = cv2.SimpleBlobDetector_create(params)


######### INITIAL CALIBRATION #########

# Take first frame of video after enough time has passed
warmup_counter = 0
while warmup_counter < 40:
	warmup_counter += 1
	ret, im = cap.read()
ret, im = cap.read()
im = im[10:710, 300:950]



# Detect the blobs in the initial position, and draw them
keypoints = detector.detect(im)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Get all keypoints in order (label them 1 - 69)
vertical_keypoint_sorted = sorted(keypoints, key=lambda x: x.pt[1])

point_counter = 0
keypoint_matrix = []
for row_size in [5, 7, 9, 9, 9, 9, 9, 7, 5]:
	row_to_sort = vertical_keypoint_sorted[point_counter:point_counter + row_size]
	row_to_sort.sort(key=lambda x: x.pt[0])
	point_counter = point_counter + row_size

	keypoint_row = []
	for point in row_to_sort:
		new_keypoint = KeypointClass(point)
		keypoint_row.append(new_keypoint)

	for num_zeroes in range((9 - row_size)/2):
		keypoint_row.insert(0, None)
		keypoint_row.append(None)

	keypoint_matrix.append(keypoint_row) # Add new row to the keypoint matrix


# Set the params from calibration
params_a = [1.123, 1.2484, 1.3275, 1.13152, 1.0722, 1.1918, 1.4361, 1.4357, 1.6245, 1.4751, 1.195, 1.3036, 1.2359, 1.483, 1.5648, 1.5525, 1.5075, 1.5278, 1.524, 1.4042, 1.2838, 1.2527, 1.549, 1.5131, 1.493, 1.4872, 1.6935, 1.5316, 1.5548, 1.4444, 1.5215, 1.5695, 1.517, 1.5472, 1.5531, 1.4687, 1.4713, 1.437, 1.4868, 1.3121, 1.539, 1.539, 1.5753, 1.6508, 1.5535, 1.4167, 1.5071, 1.4955, 1.0824, 1.4406, 1.4781, 1.5627, 1.5825, 1.4875, 1.5425, 1.3967, 1.3707, 1.4233, 1.4663, 1.5173, 1.4514, 1.479, 1.5723, 1.3656, 1.5673, 1.588, 1.6008, 1.5734, 1.4203]
params_b = [24.903, 23.74, 24.411, 24.008, 23.891, 24.646, 24.808, 24.978, 24.377, 24.124, 24.769, 24.892, 26.608, 24.781, 24.366, 24.422, 24.344, 23.858, 24.377, 25.01, 24.427, 24.439, 24.627, 24.814, 23.987, 25.083, 24.791, 24.082, 24.575, 25.036, 24.278, 24.557, 24.958, 24.383, 24.364, 23.921, 24.222, 24.383, 24.893, 24.839, 25.008, 24.328, 23.651, 24.109, 24.429, 24.479, 24.27, 25.684, 24.432, 24.743, 25.0, 24.137, 24.353, 24.929, 23.883, 24.243, 24.415, 24.811, 24.273, 24.327, 24.321, 23.689, 24.734, 24.708, 25.786, 25.433, 25.226, 24.981, 25.12]

for l in range(len(params_a)):
	(i, j) = index_to_ij(l)
	keypoint_matrix[i][j].set_params(params_a[l], params_b[l])


# Set the origin point now that we have the centralized dot
ORIGIN_POINT = keypoint_matrix[4][4].point.pt

xs = [] 
ys = []
zs = []
colors = []

# Now that we have constructed the keypoint_matrix, we can do things with the points (initialize)
for i, row in enumerate(keypoint_matrix):
	for j, point in enumerate(row):
		if not point:
			continue

		point.update_initial_pos()


		# Get original distances to neighbours
		h_n = HorNeighIndices(i, j)	
		v_n = VerNeighIndices(i, j)

		if len(h_n) > 0: # Then the point has both horizontal neighbours
			h0 = keypoint_matrix[h_n[0][0]][h_n[0][1]]
			h1 = keypoint_matrix[h_n[1][0]][h_n[1][1]]
			point.set_hor_dists(xy_distance(point, h0), xy_distance(point, h1))

		if len(v_n) > 0: # Then the point has both vertical neighbours
			v0 = keypoint_matrix[v_n[0][0]][v_n[0][1]]
			v1 = keypoint_matrix[v_n[1][0]][v_n[1][1]]
			point.set_ver_dists(xy_distance(point, v0), xy_distance(point, v1))


		# Label the point
		# cv2.putText(im_with_keypoints, "(%s,%s)" % (i, j), (int(point.x()), int(point.y())), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 102, 0))

		# Get real coordinates of keypoint
		coords = point.pos_3D()
		xs.append(coords[0])
		ys.append(coords[1])

		
		if np.isnan(point.z_offset):
			zs.append(coords[2])
			# print np.sqrt(HEMISPHERE_RADIUS**2 - coords[0]**2 - coords[1]**2)
			print np.sqrt(coords[0]**2 + coords[1]**2) 
			colors.append('red')
		else:
			zs.append(coords[2] + point.z_offset)
			# print coords[2] +  point.z_offset
			colors.append('blue')
	

######### SHOW KEYPOINTS #########
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


######### VIEW CALIBRATION PLOT #########
'''
fig = plt.figure()
plt.axis('equal')

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=colors)

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5,5)
ax.set_zlim3d(0, 5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
'''


######### ADD THE MULTITRACKERS #########
tracker_type = 'KCF'
tracker = cv2.MultiTracker_create()

for i, row in enumerate(keypoint_matrix):
	for j, point in enumerate(row):
		if not point:
			continue

		# For closer up
		r, h, c, w = int(point.y()-point.r() - 3), \
			2 * int(point.r()) + 15, \
			int(point.x() - point.r() - 3), \
			2 * int(point.r()) + 15

		# For farther away
		# r, h, c, w = int(point.y()-point.r() + 1), \
		# 	2 * int(point.r()) + 18, \
		# 	int(point.x() - point.r() - 3), \
		# 	2 * int(point.r()) + 13

		bbox = (c, r, w, h)
		tracker.add(cv2.TrackerKCF_create(), im, bbox)


# while(1):

def update_graph(num):
	ok, frame = cap.read()
	frame = frame[10:710, 300:950]
	im_with_keypoints = frame
	if ok == True:
		timer = cv2.getTickCount()

		ok, boxes = tracker.update(frame)

		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

		# Draw bounding box
		if ok:

			all_keypoints = []
			im_with_keypoints = cv2.drawKeypoints(frame, all_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

			for index, bbox in enumerate(boxes):
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(im_with_keypoints, p1, p2, (255,0,0), 2, 1)
				
				c, r, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
				tracked_roi = frame[r:r+h, c:c+w]
				# cv2.imshow("roi", tracked_roi)

				# Use blob detection
				keypoints = detector.detect(tracked_roi)

				(i, j) = index_to_ij(index)

				for keypoint in keypoints:
					shifted_pt = (keypoint.pt[0] + c, keypoint.pt[1] +  r)
					keypoint.pt = shifted_pt
					all_keypoints.append(keypoint)
					keypoint_matrix[i][j].point = keypoint


				# cv2.putText(im_with_keypoints, "(%s, %s)" % (i, j), (c, r), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 102, 0))

			im_with_keypoints = cv2.drawKeypoints(frame, all_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


				
		else :
			# Tracking failure
			pass
			# cv2.putText(im_with_keypoints, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
		# Display tracker type on frame
		# cv2.putText(im_with_keypoints, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	 
		# Display FPS on frame
		cv2.putText(im_with_keypoints, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
		# Display result
		# cv2.imshow("Tracking", frame)
		cv2.imshow("Keypoints", im_with_keypoints)

		xs = []
		ys = []
		zs = []
		colors = []
		for i, row in enumerate(keypoint_matrix):
			for j, point in enumerate(row):
				if not point:
					continue
				coords = point.shifted_z_pos_3D()



				point.add_size_to_history()

				h_n = HorNeighIndices(i, j)	
				v_n = VerNeighIndices(i, j)

				is_contact = False

				# Check if the neighbour distances have increased
				if len(h_n) > 0: # Then the point has two horizontal neighbours
					h0 = keypoint_matrix[h_n[0][0]][h_n[0][1]]
					h1 = keypoint_matrix[h_n[1][0]][h_n[1][1]]

					# Check if the horizontal distances have increased
					if xy_distance(point, h0) > 1.13 * point.hor_dists[0] or xy_distance(point, h1) > 1.13 * point.hor_dists[1]:
						is_contact = True

				if len(v_n) > 0: # Then the point has two vertical neighbours
					v0 = keypoint_matrix[v_n[0][0]][v_n[0][1]]
					v1 = keypoint_matrix[v_n[1][0]][v_n[1][1]]
					
					if xy_distance(point, v0) > 1.13 * point.ver_dists[0] or xy_distance(point, v1) > 1.13 * point.ver_dists[1]:
						is_contact = True


				if is_contact:
					colors.append('red')
				else:
					colors.append('white')

				# if len(h_n) > 0 and len(v_n) > 0:
				# 	n1 = keypoint_matrix[h_n[0][0]][h_n[0][1]]
				# 	n2 = keypoint_matrix[h_n[1][0]][h_n[1][1]]
				# 	m1 = keypoint_matrix[v_n[0][0]][v_n[0][1]]
				# 	m2 = keypoint_matrix[v_n[1][0]][v_n[1][1]]

				# 	if n1 and n2 and m1 and m2:

				# 		# print np.sqrt((n1.pos_3D()[1] - point.pos_3D()[1])**2 + (n1.pos_3D()[0] - point.pos_3D()[0])**2)
				# 		# print np.sqrt((m1.pos_3D()[1] - point.pos_3D()[1])**2 + (m1.pos_3D()[0] - point.pos_3D()[0])**2)
				# 		# print np.sqrt((m2.pos_3D()[1] - point.pos_3D()[1])**2 + (m2.pos_3D()[0] - point.pos_3D()[0])**2)
				# 		# print np.sqrt((n2.pos_3D()[1] - point.pos_3D()[1])**2 + (n2.pos_3D()[0] - point.pos_3D()[0])**2)

				# 		[R1, P1] = circle_from_three_points(n1, point, n2)
				# 		[R2, P2] = circle_from_three_points(m1, point, m2)

				# 		if (np.sqrt(P1[0]**2 + P1[1]**2 + P1[2]**2) > np.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) and R1 < 8) \
				# 		or (np.sqrt(P2[0]**2 + P2[1]**2 + P2[2]**2) > np.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) and R2 < 8) \
				# 		:
				# 			colors.append('red')
				# 		else:
				# 			colors.append('blue')
				# 	else:
				# 		colors.append('blue')
				# else:
				# 	colors.append('blue')

				# Add point's coordinates to the 3D plot
				xs.append(coords[0])
				ys.append(coords[1])

				
 				if is_contact:
 					if not np.isnan(point.actual_z) and coords[2] > point.actual_z:
 						zs.append(point.actual_z)
					else:
						zs.append(coords[2])
				else:
					zs.append(point.actual_z)
					# zs.append(coords[2])
				

				# zs.append(coords[2])

		# graph.set_data(xs, ys)
		# graph.set_3d_properties(zs)
		graph._offsets3d = (xs, ys, zs)

		graph._facecolor3d = colors
		graph._edgecolor3d = 'grey'
		return title, graph, 





# cv2.waitKey()
# cv2.destroyAllWindows()
# cap.release()



a = np.random.rand(2000, 3)*10
t = np.array([np.ones(100)*i for i in range(20)]).flatten()
df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})


# When everything done, release the capture

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Membrane Reconstruction')
ax.set_xlim3d([-5, 5])
ax.set_xlabel('X')

ax.set_ylim3d([-5, 5])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 5])
ax.set_zlabel('Z')
ax.view_init(azim=90, elev=90)


data=df[df['time']==0]



# graph, = ax.plot(data.x, data.y, data.z, linestyle="", marker="o")
graph = ax.scatter(data.x, data.y, data.z, marker="o")


ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, 
							   interval=40, blit=False)

plt.show()


for i, row in enumerate(keypoint_matrix):
	for j, point in enumerate(row):
		if not point:
			continue
		# print "[%s, %s]: %s" % (i, j, np.var(point.get_size_history()))
		print "%s" % (point.size())
