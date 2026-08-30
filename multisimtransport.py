# python

# standard modules
import csv, math, os, sys
from copy import deepcopy

# math & plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi
from numpy.lib.stride_tricks import sliding_window_view

# custom modules
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FOLDER_PATH)

from network.numpyrbf import RBFNetwork, load_model

# Python handles for CoppeliaSim
try:
	import sim
except ImportError:
	pass

# ================================================================================
# PARAMETERS & CONFIGURATION (DEFAULTS - OVERRIDDEN IF PASSED FROM RUNNER)
# ================================================================================
SCHEME = globals().get("SCHEME", "adversarial_cooperation")  # "kuramoto" vs "adversarial_cooperation"
DIRECTION = globals().get("DIRECTION", 1)
ETA = globals().get("ETA", 500)
OMEGA = globals().get("OMEGA", 1)
DATA_COLLECT = globals().get("DATA_COLLECT", 1)
PLOT = globals().get("PLOT", 0)
LOAD = globals().get("LOAD", 1)
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshot")

# External Parameters for Team Sizes
NPOSITIVE = globals().get("NPOSITIVE", 2)
NNEGATIVE = globals().get("NNEGATIVE", 2)

# Dynamic variables (Initialized on script load, re-computed in sysCall_init)
NUM_ROBOTS = NPOSITIVE + NNEGATIVE
print(NUM_ROBOTS)
NAMESPACES = [f"rob{i+1}" for i in range(NUM_ROBOTS)]

TURNING_GAINS = np.ones(NUM_ROBOTS) * 0.2
TURNING_DGAINS = 0.7
TEGOTAE_CLIP = 0.5
TARGET_HZ = 20
DT_TARGET = 1.0 / TARGET_HZ
N_HISTORY = 200
ACCEPTANCE_RADIUS = 1.5
SKIP = 20

# ================================================================================
# DYNAMIC MATRIX GENERATION
# ================================================================================
def build_centralized_coupling_matrices(n_pos, n_neg, direction):
	"""Generates K_CENTRALIZE and O_CENTRALIZE dynamically based on team sizes."""
	total = n_pos + n_neg
	k_mat = np.zeros((total, total))
	
	for i in range(total):
		for j in range(total):
			if i == j:
				continue
			is_i_pos = i < n_pos
			is_j_pos = j < n_pos
			
			if is_i_pos == is_j_pos:
				same_team_count = n_pos if is_i_pos else n_neg
				k_mat[i, j] = 0.5 / max(1, same_team_count - 1)
			else:
				other_team_count = n_neg if is_i_pos else n_pos
				k_mat[i, j] = 0.25 / max(1, other_team_count)

	phase_to_win = -np.pi / 2.0 if direction > 0 else np.pi / 2.0
	o_mat = np.zeros((total, total))
	
	o_mat[:n_pos, n_pos:] = -phase_to_win
	o_mat[n_pos:, :n_pos] = phase_to_win
	
	return k_mat, o_mat

K_CENTRALIZE, O_CENTRALIZE = build_centralized_coupling_matrices(NPOSITIVE, NNEGATIVE, DIRECTION)

# ================================================================================
# GLOBAL STATE
# ================================================================================
ti = 0
printi = 0
goalstages = 0
goallist = []
forward_models = []
phis = np.zeros(NUM_ROBOTS)
tegotae = np.zeros(NUM_ROBOTS)
tegotae_min = np.zeros(NUM_ROBOTS) + 0.0
tegotae_max = np.zeros(NUM_ROBOTS) + 0.0
tegotae_mean = np.zeros(NUM_ROBOTS)
tegotae_windup = np.zeros(NUM_ROBOTS)
yaws = np.zeros((NUM_ROBOTS,))
positive_gain = np.zeros((NUM_ROBOTS,))
negative_gain = np.zeros((NUM_ROBOTS,))
current_history = np.zeros((400, NUM_ROBOTS))
currents_offset = np.zeros(NUM_ROBOTS)
motors = {}
robots = {}
rob_dummies = {}
car_dummies = {}
cargo = None
goal = None
arrayhistory = []
graph_handle = None
phi_stream_handles = []
capsule_handles = []

def refresh_dynamic_parameters():
	"""Recalculates dynamic arrays and sizes if variables were passed in init_globals."""
	global NUM_ROBOTS, NAMESPACES, TURNING_GAINS, K_CENTRALIZE, O_CENTRALIZE
	global phis, tegotae, tegotae_min, tegotae_max, tegotae_mean, tegotae_windup
	global yaws, positive_gain, negative_gain, current_history, currents_offset

	NUM_ROBOTS = NPOSITIVE + NNEGATIVE
	NAMESPACES = [f"rob{i+1}" for i in range(NUM_ROBOTS)]
	TURNING_GAINS = np.ones(NUM_ROBOTS) * 0.2
	print(NAMESPACES)
	K_CENTRALIZE, O_CENTRALIZE = build_centralized_coupling_matrices(NPOSITIVE, NNEGATIVE, DIRECTION)

	# python
	
	phis = np.random.uniform(-np.pi / 2, np.pi, NUM_ROBOTS)
	tegotae = np.zeros(NUM_ROBOTS)
	tegotae_min = np.zeros(NUM_ROBOTS)
	tegotae_max = np.zeros(NUM_ROBOTS)
	tegotae_mean = np.zeros(NUM_ROBOTS)
	tegotae_windup = np.zeros(NUM_ROBOTS)
	yaws = np.zeros((NUM_ROBOTS,))
	positive_gain = np.zeros((NUM_ROBOTS,))
	negative_gain = np.zeros((NUM_ROBOTS,))
	current_history = np.zeros((400, NUM_ROBOTS))
	currents_offset = np.zeros(NUM_ROBOTS)

# ================================================================================
# FUNCTION: sysCall_init
# ================================================================================
def sysCall_init():
	global phis, currents_offset
	global cargo, goal, rob_dummies, car_dummies, goallist
	global graph_handle, phi_stream_handles
	global positive_gain, negative_gain, forward_models
	global line_drawing_handle

	refresh_dynamic_parameters()

	# Motor and robot handles
	for i, name in enumerate(NAMESPACES):
		lmotor = sim.getObject(f"/{name}/lmotor")
		rmotor = sim.getObject(f"/{name}/rmotor")

		motors[name] = (lmotor, rmotor)
		robots[name] = sim.getObject(f"/{name}")
		rob_dummies[name] = sim.getObject(f"/{name}/robo")
		car_dummies[name] = sim.getObject(f"/cargo_{i + 1}")

	cargo = sim.getObject(f"/cargo")
	goal = sim.getObject(f"/goal")
	#goallist = [sim.getObject(f"/subgoal{str(i)}") for i in range(1,15,1)]
	goallist = goallist + [goal]

	if SCHEME == "adversarial_cooperation":
		color_prefix = "blue" if DIRECTION >= 1 else "red"
		data_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}win_nofilter.csv")
		training_data = pd.read_csv(data_path).to_numpy()
		
		if LOAD:
			load_rbf_models(training_data)
		else:
			train_rbf_models(training_data)

		for i in range(NUM_ROBOTS):
			robid = 0 if i < NPOSITIVE else 3
			grad = forward_models[i].backward_phi(training_data[:, [robid]], training_data[:, [robid + 4]])
			positive_gain[i] = (grad[grad > 0].mean())
			negative_gain[i] = np.abs(grad[grad < 0].mean())

		if PLOT:
			plot_training_results(training_data)



	

	# Graph handles
	graph_handle = sim.getObject("/Graph")
	init_line_capsules(count=NUM_ROBOTS)

	# python
	# Assign Blue to Positive team, Red to Negative team
	blue_color = [0.12, 0.47, 0.71]  # Tab10 Blue
	red_color = [0.84, 0.15, 0.16]   # Tab10 Red

	colors = [blue_color if i < NPOSITIVE else red_color for i in range(NUM_ROBOTS)]

	for i in range(NUM_ROBOTS):
		stream_id = sim.addGraphStream(graph_handle, f"T{i}", "Nm", 0, colors[i])
		phi_stream_handles.append(int(stream_id))

# ================================================================================
# FUNCTION: sysCall_actuation
# ================================================================================
def sysCall_actuation():
	global ti, printi, SKIP
	global phis, tegotae, arrayhistory, goal, rob_dummies, car_dummies, goallist, goalstages
	global yaws, tegotae_windup, graph_handle, phi_stream_handles

	dt = DT_TARGET
	goalobj = goallist[goalstages]

	# Step 0: Sensing
	cargo_pose = sim.getObjectPosition(cargo, sim.handle_world)
	goal_pose = sim.getObjectPosition(goalobj, sim.handle_world)
	cargo_orien = sim.getObjectOrientation(cargo, sim.handle_world)
	currents = get_estForce()
	robot_poses, yaws = get_robPoses()
	robot_dyaws = get_robVels()

	distance_to_goal = np.linalg.norm(np.array(goal_pose[:-1]) - np.array(cargo_pose[:-1]), axis=-1)
	# Inside your sysCall_actuation() loop:
	if distance_to_goal < ACCEPTANCE_RADIUS:
		if goalstages < len(goallist)-1:
			goalstages += 1

	if DATA_COLLECT > 0:
		if (int(printi) % 300 == 0):

			#init_line_capsules(count=NUM_ROBOTS)
			update_robot_cargo_lines()

			frame_idx = int(printi // 300)
			if (printi != 0):
				take_snapshot(camera_name="snapshot", filename=f"frame_{frame_idx}.png")
			#clear_capsule()
			print(distance_to_goal)

	# Step 1: Advance phase baseline
	phis = (phis + OMEGA * dt) % (2.0 * np.pi)

	# Step 2: Compute Coupling Feedback
	if SCHEME == "kuramoto":
		tegotae = compute_centralized_coupling()
	elif SCHEME == "adversarial_cooperation":
		tegotae = compute_decentralized_coupling(currents)
	
	# Step 3: Update phase angles & force
	phis = (phis + dt * tegotae) % (2.0 * np.pi)
	force = np.clip(np.sin(phis), 0.0, 1.0)
	
	# Step 4: Compute turning & drive motors
	turning = compute_turning(goal_pose, robot_poses, yaws, robot_dyaws, force)
	left_forces = 2 * 0.5 * force * np.clip(1.0 + force * turning, -1, 1) 
	right_forces = 2 * 0.5 * force * np.clip(1.0 - force * turning, -1, 1) 
	
	for idx, name in enumerate(NAMESPACES):
		drive(name, float(left_forces[idx]), float(right_forces[idx]), turning=force[idx] * turning[idx])

	# Step 5: Recording & Visualization
	if DATA_COLLECT:
		data = np.concatenate([
			np.array([ti]),
			cargo_pose,
			cargo_orien,
			phis.flatten(),
			currents.flatten(),
			robot_poses.flatten(),
			yaws.flatten()
		])
		record(data, skip=SKIP)

	for i, stream_id in enumerate(phi_stream_handles):
		sim.setGraphStreamValue(graph_handle, stream_id, force[i])

	ti += 1.0 / TARGET_HZ
	printi += 1.0

# ================================================================================
# FUNCTION: Supplementary Helper Logic
# ================================================================================
def compute_centralized_coupling():
	delta_phi = (phis[:, None] + O_CENTRALIZE) - phis[None, :]
	return np.sum(K_CENTRALIZE * np.sin(delta_phi), axis=0)

def compute_decentralized_coupling(currents):
	global tegotae_windup

	tegotae_raw = np.zeros(NUM_ROBOTS)
	for i in range(NUM_ROBOTS):
		adjusted_current = np.clip(currents[i], 0, None)

		for j, gamma in zip([i], [1]):
			grad_phi_fine = forward_models[j].backward_phi(phis[i], adjusted_current)
			grad_phi_fine[grad_phi_fine > 0] *= 0.003 / positive_gain[j]
			grad_phi_fine[grad_phi_fine < 0] *= 0.003 / negative_gain[j]
			
			tegotae_max[i] = 0.995 * tegotae_max[i] + 0.005 * grad_phi_fine
			grad_phi_fine -= 0.2 * tegotae_max[i]

			grad_phi = gamma * grad_phi_fine
			tegotae_raw[i] += grad_phi

	if ti < 15:
		tegotae_raw *= 0.0

	raw_drive = ETA * tegotae_raw
	total_drive = raw_drive + tegotae_windup
	tegotae_clipped = np.clip(total_drive, -TEGOTAE_CLIP, TEGOTAE_CLIP)
	
	tegotae_windup += (raw_drive - tegotae_clipped)
	return tegotae_clipped

def generate_side_signs(n, m):
	total = n + m
	
	# 1s on each outer end (half of the remaining positive slots)
	outer_ones_count = (total - (n + m - 2)) // 2  # standard 1 on each end or scaled
	left_ones = max(1, n // 2 - 1) if n > 3 else 1
	right_ones = max(1, m // 2) if m > 3 else 1
	
	# Count of interior negative numbers
	neg_count = total - (left_ones + right_ones + 2)
	
	# Build the full sequence
	side_signs = np.concatenate([
		np.ones(left_ones, dtype=int),
		[0],
		-np.ones(neg_count, dtype=int),
		[0],
		np.ones(right_ones, dtype=int)
	])
	
	return side_signs

def compute_turning(goal_pose, robot_poses, yaws, robot_dyaws, force, offset=True):
	global TURNING_DGAINS, TURNING_GAINS
	global NPOSITIVE,NNEGATIVE

	yaws_arr = np.array(yaws)
	
	if offset:
		side_signs = np.array(np.zeros(NPOSITIVE+NNEGATIVE,))
		#side_signs = np.array([1,-1,1,-1])
		robotwise_goal = compute_tangent_points(goal_pose[:-1], robot_poses, side_signs)
	else:
		robotwise_goal = np.expand_dims(goal_pose[:-1], axis=0)

	distance_to_goal = robotwise_goal - robot_poses
	theta = 0*np.arctan2(distance_to_goal[:, 1], distance_to_goal[:, 0])

	if DIRECTION > 0:
		theta[NPOSITIVE:] += np.pi
	elif DIRECTION < 0:
		theta[:NPOSITIVE] += np.pi

	turning = np.sin(theta) * np.cos(yaws_arr) - np.cos(theta) * np.sin(yaws_arr)
	turning = -TURNING_GAINS * np.clip(np.sin(turning) * 3, -1, 1)
	turning += (
		TURNING_DGAINS
		* robot_dyaws
		* ((turning + TURNING_DGAINS * robot_dyaws) * turning > 0)
	)

	return turning

def record(list_of_data, skip=1):
	global arrayhistory

	hist = []
	for data in [list_of_data]:
		if isinstance(data, np.ndarray):
			hist += data.flatten().tolist()
		elif isinstance(data, list):
			hist += np.array(data).flatten().tolist()

	arrayhistory.append(hist)

	if len(arrayhistory) > N_HISTORY:
		arrayhistory = arrayhistory[::skip]
		csv_path = os.path.join(FOLDER_PATH, "data/data.csv")
		with open(csv_path, mode="a", newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(arrayhistory)

		arrayhistory = []

def train_rbf_models(training_data):
	for i in range(NUM_ROBOTS):
		swap_idx = (i + NPOSITIVE) % NUM_ROBOTS
		training_x = np.concatenate([np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])], axis=-1)
		training_x_ = np.concatenate([np.sin(training_data[:, [swap_idx]]), np.cos(training_data[:, [swap_idx]])], axis=-1)
		
		training_y = training_data[:, [NUM_ROBOTS + i]]
		padded_y = np.pad(training_y, pad_width=((400 - 1, 0), (0, 0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400, axis=0)
		training_y = training_y - np.mean(windows_np, axis=-1)

		training_y_ = training_data[:, [NUM_ROBOTS + swap_idx]]
		padded_y = np.pad(training_y_, pad_width=((400 - 1, 0), (0, 0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400, axis=0)
		training_y_ = training_y_ - np.mean(windows_np, axis=-1)

		training_x = np.concatenate([training_x, training_x_], axis=0)
		training_y = np.concatenate([training_y, training_y_], axis=0)

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		model_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{i + 1}.npz")

		model = RBFNetwork(n_centers=10, sigma=1.5, lr_centers=0.1, lr_weights=0.1)
		print(f"Training Robot {i + 1} | X: {training_x.shape}, Y: {training_y.shape}")
		model.fit(training_x, training_y, verbose=0, epochs=3000)
		model.save(model_path)

		forward_models.append(model)

def load_rbf_models(training_data):
	for i in range(NUM_ROBOTS):

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		robid = 0 if i < NPOSITIVE else 3

		training_y = training_data[:, robid]
		currents_offset[i] = np.min(training_y)

		model_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{robid+1}.npz")

		model = load_model(model_path)
		forward_models.append(model)

def plot_training_results(training_data):
	rows = int(np.ceil(NUM_ROBOTS / 2.0))
	fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
	axes = axes.flatten()

	for i in range(NUM_ROBOTS):
		swap_idx = (i + NPOSITIVE) % NUM_ROBOTS
		training_x = np.concatenate([np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])], axis=-1)
		training_y = training_data[:, [NUM_ROBOTS + i]]
		padded_y = np.pad(training_y, pad_width=((400 - 1, 0), (0, 0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400, axis=0)
		training_y = training_y - np.mean(windows_np, axis=-1)

		predictions = forward_models[i].predict(training_x)
		predictions_ = forward_models[swap_idx].predict(training_x)

		ax = axes[i]
		ax.plot(predictions_[:], label="Predictions_", linestyle="-", c='tab:red')
		ax.plot(predictions[:], label="Predictions", linestyle="-", c='tab:orange')
		ax.plot(training_y[:], label="True Y", alpha=0.7, linestyle="--", c='k')

		ax.set_title(f"Namespace: {NAMESPACES[i]}")
		ax.legend()

	plt.tight_layout()
	plt.show()

def drive(name, left_forces, right_forces, turning=0):
	global motors

	lmotor, rmotor = motors[name]

	sim.setJointTargetForce(lmotor, left_forces)
	sim.setJointTargetForce(rmotor, right_forces)

	turninggain = 0.5
	vl = 1 + np.clip(2 * turninggain * turning, -2, 2)
	vr = 1 + np.clip(-2 * turninggain * turning, -2, 2)

	sim.setJointTargetVelocity(lmotor, 2 * vl)
	sim.setJointTargetVelocity(rmotor, 2 * vr)

def get_estForce():
	global NAMESPACES, rob_dummies, car_dummies

	currents = np.zeros(NUM_ROBOTS)
	current_history[:-1] = current_history[1:]

	for i, name in enumerate(NAMESPACES):
		dumA = np.array(sim.getObjectPosition(rob_dummies[name], sim.handle_world))
		dumB = np.array(sim.getObjectPosition(car_dummies[name], sim.handle_world))

		if np.abs(current_history[:, i]).sum() == 0:
			current_history[:, i] = np.linalg.norm(dumA - dumB, axis=-1)
		else:
			current_history[-1, i] = np.linalg.norm(dumA - dumB, axis=-1)

		currents[i] = np.clip(np.linalg.norm(dumA - dumB, axis=-1) - np.percentile(current_history[:, i], 50), 0, None)
	return currents

def get_robPoses():
	global NAMESPACES, robots

	robot_poses = np.zeros((NUM_ROBOTS, 2))
	yaws = np.zeros(NUM_ROBOTS)
	for i, name in enumerate(NAMESPACES):
		ori = sim.getObjectOrientation(robots[name], sim.handle_world)
		pos = sim.getObjectPosition(robots[name], sim.handle_world)

		robot_poses[i] = pos[:-1]
		yaws[i] = ori[2]
	return robot_poses, yaws

def get_robVels():
	global NAMESPACES, robots

	dyaws = np.zeros(NUM_ROBOTS)
	for i, name in enumerate(NAMESPACES):
		linearVelocity, angularVelocity = sim.getObjectVelocity(robots[name])
		dyaws[i] = angularVelocity[2]
	return dyaws

def compute_tangent_points(center: np.ndarray, points: np.ndarray, side_sign: np.ndarray, r=0.3) -> np.ndarray:
	delta = center - points
	D = np.hypot(delta[:, 0], delta[:, 1])
	L = np.sqrt(D**2 - r**2)

	alpha = np.arctan2(delta[:, 1], delta[:, 0])
	phi = np.arcsin(r / D)

	theta = alpha + side_sign * phi

	x_prime = points[:, 0] + L * np.cos(theta)
	y_prime = points[:, 1] + L * np.sin(theta)

	return np.column_stack((x_prime, y_prime))

def init_line_capsules(count=NUM_ROBOTS, radius=0.015, color=[0.8, 0.8, 0.8]):
	global capsule_handles
	capsule_handles = []

	for _ in range(count):
		# 1. Create cylinder primitive
		handle = sim.createPrimitiveShape(
			sim.primitiveshape_cylinder, [radius * 2.0, radius * 2.0, 1.0], 0
		)
		
		# 2. Set physics parameters (non-respondable, static/non-dynamic)
		sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, 0)
		sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)

		# 3. Set visual color
		sim.setShapeColor(
			handle, None, sim.colorcomponent_ambient_diffuse, color
		)

		# 4. Hide below ground level (-100m)
		sim.setObjectPosition(handle, sim.handle_world, [0.0, 0.0, -100.0])

		capsule_handles.append(handle)


def update_robot_cargo_lines():
	global capsule_handles

	for i, name in enumerate(NAMESPACES):
		if i >= len(capsule_handles):
			break


		handle = capsule_handles[i]


		if 1:
			rob_dummy = sim.getObject(f"/{name}/robo")
			car_dummy = (
				car_dummies[name]
				if name in car_dummies
				else sim.getObject(f"/cargo_{i+1}")
			)

			p1 = np.array(sim.getObjectPosition(rob_dummy, sim.handle_world), dtype=float)
			p2 = np.array(sim.getObjectPosition(car_dummy, sim.handle_world), dtype=float)

			vec = p2 - p1
			target_length = float(np.linalg.norm(vec))

			if target_length < 1e-4:
				continue

			# 1. Read the current bounding box dimensions [size_x, size_y, size_z]
			_, _, bb_size = sim.getShapeBB(handle)
			
			# 2. Get current length along the Z-axis (capsules in CoppeliaSim are aligned along Z by default)
			
			current_length = bb_size

			if current_length > 1e-4:
				# 3. Calculate scale factor relative to current length
				scale_factor = target_length / current_length
				
				# 4. Scale object (keep X and Y scale at 1.0, scale Z dynamically)
				sim.scaleObject(handle, 1.0, 1.0, scale_factor)

			midpoint = (p1 + p2) / 2.0
			pos_list = [
				float(midpoint[0]),
				float(midpoint[1]),
				float(midpoint[2])-0.1,
			]
			sim.setObjectPosition(handle, -1, pos_list)
			
			# 2. Normalized direction vector (from p1 to p2)
			dir_vec = vec / target_length

			# 3. Default orientation of a CoppeliaSim capsule points along local Z
			v_from = np.array([0.0, 0.0, 1.0])
			v_to = dir_vec

			# 4. Calculate Quaternion [x, y, z, w] to rotate v_from onto v_to
			dot = float(np.dot(v_from, v_to))

			if dot > 0.999999:
				# Already aligned along +Z
				quat = [0.0, 0.0, 0.0, 1.0]
			elif dot < -0.999999:
				# Directly opposite (-Z), rotate 180 deg around X-axis
				quat = [1.0, 0.0, 0.0, 0.0]
			else:
				# Cross product gives the rotation axis
				axis = np.cross(v_from, v_to)
				
				# Quaternion vector component (x, y, z) and scalar component (w)
				# Using the half-angle formula for unit vectors:
				w = 1.0 + dot
				quat_vec = axis
				
				# Construct unnormalized [x, y, z, w]
				quat_raw = np.array([quat_vec[0], quat_vec[1], quat_vec[2], w])
				
				# Normalize the quaternion
				quat_norm = quat_raw / np.linalg.norm(quat_raw)
				quat = quat_norm.tolist()

			# 5. Set orientation in CoppeliaSim using Quaternion [x, y, z, w]
			sim.setObjectQuaternion(handle, sim.handle_world, quat)



def clear_capsule():
	global capsule_handles
	
	for handle in capsule_handles:
		try:
			sim.removeObject(handle)
		except Exception:
			pass
	capsule_handles.clear()


def take_snapshot(camera_name="snapshot", filename="capture.png"):
	if not os.path.exists(SNAPSHOT_DIR):
		os.makedirs(SNAPSHOT_DIR, exist_ok=True)

	try:
		cam_handle = sim.getObject(f"/{camera_name}")

		try:
			sim.handleVisionSensor(cam_handle)
		except Exception:
			pass

		image_buffer, resolution = sim.getVisionSensorImg(cam_handle)

		if image_buffer:
			img = np.frombuffer(image_buffer, dtype=np.uint8)
			img = img.reshape((resolution[1], resolution[0], 3))
			img = np.flipud(img)

			save_path = os.path.join(SNAPSHOT_DIR, filename)
			plt.imsave(save_path, img)
			print(f"[Snapshot] Saved image to: {save_path}")

	except Exception as e:
		print(f"[Snapshot Error] Failed to capture snapshot: {e}")
# ================================================================================
# FUNCTION: sysCall_cleanup
# ================================================================================
def sysCall_cleanup():

	global capsule_handles

	for name in NAMESPACES:
		if name in motors:
			lmotor, rmotor = motors[name]
			sim.setJointTargetForce(lmotor, 0.0)
			sim.setJointTargetForce(rmotor, 0.0)

	clear_capsule()
	