#python

# standard modules
import csv, math, os, sys
from copy import deepcopy

# math & plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# CONSTANTS & CONFIGURATION
# ================================================================================
SCHEME = "decentralized" # "centralized" vs "decentralized"
DIRECTION = 1
ETA = 3
DATA_COLLECT = 0
PLOT = 0
LOAD = 1

NAMESPACES = ["rob1", "rob2", "rob3", "rob4"]
TURNING_GAINS = np.array([1.0, 1.0, 1.0, 1.0])*0.2
TURNING_DGAINS = 0.7
TEGOTAE_CLIP = 0.2
TARGET_HZ = 20
DT_TARGET = 1.0 / TARGET_HZ
N_HISTORY = 200

# Phase offset for winning/losing team coupling

K_CENTRALIZE = np.array(
	[
		[0.0, 0.5, 0.25, 0.25],  # Robot 0
		[0.5, 0.0, 0.25, 0.25],  # Robot 1
		[0.25, 0.25, 0.0, 0.5],  # Robot 2
		[0.25, 0.25, 0.5, 0.0],  # Robot 3
	])


PHASE_TO_WIN = - np.pi / 2.0 if DIRECTION > 0 else  np.pi / 2.0
O_CENTRALIZE = np.array(
	[
		[0.0, 0.0, -PHASE_TO_WIN, -PHASE_TO_WIN],  # Robot 0 sending to others
		[0.0, 0.0, -PHASE_TO_WIN, -PHASE_TO_WIN],  # Robot 1 sending to others
		[PHASE_TO_WIN, PHASE_TO_WIN, 0.0, 0.0],  # Robot 2 sending to others
		[PHASE_TO_WIN, PHASE_TO_WIN, 0.0, 0.0],  # Robot 3 sending to others
	])

# ================================================================================
# GLOBAL STATE
# ================================================================================
forward_models = []
phis = np.zeros(len(NAMESPACES))
tegotae = np.zeros(len(NAMESPACES))
tegotae_min = np.zeros(len(NAMESPACES))
tegotae_max = np.zeros(len(NAMESPACES))
tegotae_windup = np.zeros(len(NAMESPACES))
currents_offset = np.zeros(len(NAMESPACES))
yaws = np.zeros((len(NAMESPACES),))
motors = {}
robots = {}
rob_dummies = {}
car_dummies = {}
cargo = None
goal = None
arrayhistory = []
graph_handle = None
phi_stream_handles = []

# ================================================================================
# FUNCTION: sysCall_init
# ================================================================================
def sysCall_init():
	global phis,  currents_offset
	global cargo, goal, rob_dummies, car_dummies
	global graph_handle, phi_stream_handles

	# motor and robot handles
	for i, name in enumerate(NAMESPACES):
		lmotor = sim.getObject(f"/{name}/lmotor")
		rmotor = sim.getObject(f"/{name}/rmotor")

		motors[name] = (lmotor, rmotor)
		robots[name] = sim.getObject(f"/{name}")
		rob_dummies[name] = sim.getObject(f"/{name}/robo")
		car_dummies[name] = sim.getObject(f"/cargo_{i + 1}")

	cargo = sim.getObject(f"/cargo")
	goal = sim.getObject(f"/goal")

	# Randomize initial phase angles in range [0, 2 * PI)
	phis = np.random.uniform(0.0, 2.0 * np.pi, size=4)

	if SCHEME == "decentralized":
		color_prefix = "blue" if DIRECTION >= 1 else "red"
		data_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}win_nofilter.csv")
		training_data = pd.read_csv(data_path).to_numpy()

		if LOAD: # Route to dedicated loading or training function
			load_rbf_models(training_data)
		else:
			train_rbf_models(training_data)

		
		if PLOT: # Plot predictions if enabled
			plot_training_results(training_data)


	# preparing graph
	graph_handle = sim.getObject("/Graph")

	colors = [
		[0, 0, 1],
		[0, 0.5, 1],
		[1, 0, 0],
		[1, 0.5, 0],
	]  

	for i in range(len(NAMESPACES)): # timeseries graph
		stream_id = sim.addGraphStream(graph_handle,f"T{i}","Nm",0,colors[i % len(colors)])
		phi_stream_handles.append(int(stream_id))

		


# ================================================================================
# FUNCTION: sysCall_actuation
# ================================================================================
def sysCall_actuation():
	global phis, tegotae, arrayhistory, goal, rob_dummies, car_dummies
	global yaws, tegotae_windup, graph_handle, phi_stream_handles

	dt = DT_TARGET

	# ----------------------------------------------------------------------------
	# Step 0: sensing
	# ----------------------------------------------------------------------------
	cargo_pose = sim.getObjectPosition(cargo, sim.handle_world)
	goal_pose = sim.getObjectPosition(goal, sim.handle_world)
	cargo_orien = sim.getObjectOrientation(cargo, sim.handle_world)
	currents = get_estForce()
	robot_poses, yaws = get_robPoses()
	robot_dyaws = get_robVels()
	
	# ----------------------------------------------------------------------------
	# Step 1: Advance natural phase baseline 
	# ----------------------------------------------------------------------------
	phis = (phis + 1.0 * dt) % (2.0 * np.pi)

	# ----------------------------------------------------------------------------
	# Step 2: Compute Coupling Feedback
	# ----------------------------------------------------------------------------
	if SCHEME == "centralized":
		tegotae = compute_centralized_coupling()
	elif SCHEME == "decentralized":
		tegotae = compute_decentralized_coupling(currents)
	
	# ----------------------------------------------------------------------------
	# Step 3: Update phase angles and compute instantaneous force
	# ----------------------------------------------------------------------------
	phis = (phis + dt*tegotae) % (2.0 * np.pi)
	force = np.clip(np.sin(phis), 0.0, 1.0)
	
	# ----------------------------------------------------------------------------
	# Step 4: Compute turning adjustments & apply motor forces
	# ----------------------------------------------------------------------------

	turning = compute_turning(goal_pose, robot_poses, yaws, robot_dyaws, force)

	left_forces = 0.5*force * np.clip(1.0 + force*turning,0,None)
	right_forces = 0.5*force * np.clip(1.0 - force*turning,0,None)

	# Apply target maximum forces (torque) to left and right motor joints
	for idx, name in enumerate(NAMESPACES):
		drive(name, float(left_forces[idx]),float(right_forces[idx]))

	# ----------------------------------------------------------------------------
	# Step 5: Recording & Visualization
	# ----------------------------------------------------------------------------
	if DATA_COLLECT:
		record([cargo_pose,cargo_orien,phis,currents,robot_poses,yaws])

	for i, stream_id in enumerate(phi_stream_handles):
		sim.setGraphStreamValue(graph_handle, stream_id, force[i])

# ================================================================================
# FUNCTION: supplementary
# ================================================================================

def compute_centralized_coupling():
	"""Computes Kuramoto matrix feedback for the centralized scheme."""
	delta_phi = (phis[:, None] + O_CENTRALIZE) - phis[None, :]
	return np.sum(K_CENTRALIZE * np.sin(delta_phi), axis=0)


def compute_decentralized_coupling(currents):
	"""Computes backprop Tegotae feedback and windup for the decentralized scheme."""
	global tegotae_windup

	tegotae_raw = np.zeros(len(NAMESPACES))
	for i in range(len(NAMESPACES)):
		adjusted_current = np.clip(currents[i] - currents_offset[i], 0, None)

		forward_models[i].sigma = 0.8
		grad_phi_coarse = forward_models[i].backward_phi(phis[i], adjusted_current)

		forward_models[i].sigma = 0.5
		grad_phi_fine = forward_models[i].backward_phi(phis[i], adjusted_current)

		grad_phi = 0.2 * grad_phi_coarse + 0.8 * grad_phi_fine
		tegotae_raw[i] = grad_phi / (tegotae_max[i] - tegotae_min[i])

	raw_drive = ETA * tegotae_raw
	total_drive = raw_drive + tegotae_windup
	tegotae_clipped = np.clip(total_drive, -TEGOTAE_CLIP, TEGOTAE_CLIP)
	
	# Update anti-windup accumulator
	tegotae_windup += raw_drive - tegotae_clipped
	return tegotae_clipped

def compute_turning(goal_pose, robot_poses, yaws, robot_dyaws, force):
	"""Computes target heading alignment, turning adjustments, and wheel drive forces."""
	yaws_arr = np.array(yaws)
	robotwise_goal = compute_tangent_points(
		goal_pose[:-1],
		robot_poses,
		np.array([+1, -1, -1, +1]) * np.pi / 2,
	)
	distance_to_goal = robotwise_goal - robot_poses
	theta = np.arctan2(distance_to_goal[:, 1], distance_to_goal[:, 0])

	if DIRECTION > 0:
		theta[len(NAMESPACES) // 2:] += np.pi
	elif DIRECTION < 0:
		theta[:len(NAMESPACES) // 2] += np.pi

	turning = theta - yaws_arr
	turning = -TURNING_GAINS * np.clip(np.sin(turning) * 3, -1, 1)
	turning += (
		TURNING_DGAINS
		* robot_dyaws
		* ((turning + TURNING_DGAINS * robot_dyaws) * turning > 0)
	)

	return turning

def record(list_of_data):
	"""Formats system state history and dumps to CSV when threshold is reached."""
	global arrayhistory

	hist = []
	for data in [list_of_data]:
		if isinstance(data, np.ndarray):
			hist += data.flatten().tolist()
		elif isinstance(data, list):
			hist += np.array(data).flatten().tolist()

	arrayhistory.append(hist)

	if len(arrayhistory) > N_HISTORY:
		csv_path = os.path.join(FOLDER_PATH, "data/data.csv")
		with open(csv_path, mode="a", newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(arrayhistory)

		arrayhistory = []

def train_rbf_models(training_data):
	"""Trains RBF models from scratch, saves them, and sets gradient bounds."""
	for i in range(len(NAMESPACES)):
		training_x = np.concatenate(
			[np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])],
			axis=-1,
		)
		training_y = training_data[:, [4 + i]]

		currents_offset[i] = np.min(training_y)
		training_y = np.clip(training_y - currents_offset[i], 0, None)

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		model_path = os.path.join(
			FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{i + 1}.npz"
		)

		model = RBFNetwork(n_centers=4, sigma=0.5, lr_centers=0.1, lr_weights=0.1)
		print(f"Training Robot {i + 1} | X: {training_x.shape}, Y: {training_y.shape}")
		model.fit(training_x, training_y, verbose=0, epochs=5000)
		model.save(model_path)

		forward_models.append(model)

		# Compute gradient bounds for Tegotae normalization
		grad_phi = model.backward_phi(training_data[:, [i]], 1)
		tegotae_min[i] = np.min(grad_phi)
		tegotae_max[i] = np.max(grad_phi)


def load_rbf_models(training_data):
	"""Loads pre-trained RBF models and computes gradient bounds."""
	for i in range(len(NAMESPACES)):
		training_y = training_data[:, [4 + i]]
		currents_offset[i] = np.min(training_y)

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		model_path = os.path.join(
			FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{i + 1}.npz"
		)

		model = load_model(model_path)
		forward_models.append(model)

		# Compute gradient bounds for Tegotae normalization
		grad_phi = model.backward_phi(training_data[:, [i]], 1)
		tegotae_min[i] = np.min(grad_phi)
		tegotae_max[i] = np.max(grad_phi)


def plot_training_results(training_data):
	"""Plots predictions vs true values across subplots for each robot."""
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	axes = axes.flatten()

	for i in range(len(NAMESPACES)):
		training_x = np.concatenate(
			[np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])],
			axis=-1,
		)
		training_y = np.clip(training_data[:, [4 + i]] - currents_offset[i], 0, None)
		predictions = forward_models[i].predict(training_x)

		ax = axes[i]
		ax.plot(predictions, label="Predictions", linestyle="--")
		ax.plot(training_y, label="True Y", alpha=0.7)
		ax.set_title(f"Namespace: {NAMESPACES[i]}")
		ax.legend()

	plt.tight_layout()
	plt.show()

def drive(name, left_forces, right_forces):
	global motors

	lmotor, rmotor = motors[name]

	sim.setJointTargetForce(lmotor, left_forces)
	sim.setJointTargetForce(rmotor, right_forces)



def get_estForce():
	global NAMESPACES
	global rob_dummies, car_dummies

	currents = np.zeros(len(NAMESPACES))
	for i, name in enumerate(NAMESPACES):
		dumA = np.array(sim.getObjectPosition(rob_dummies[name], sim.handle_world))
		dumB = np.array(sim.getObjectPosition(car_dummies[name], sim.handle_world))

		currents[i] =  np.linalg.norm(dumA-dumB,axis=-1)
	return currents

def get_robPoses():
	global NAMESPACES
	global robots

	robot_poses = np.zeros((len(NAMESPACES),2))
	yaws = np.zeros(len(NAMESPACES))
	for i, name in enumerate(NAMESPACES):

		ori = sim.getObjectOrientation(robots[name], sim.handle_world)  # [roll, pitch, yaw]
		pos = sim.getObjectPosition(robots[name], sim.handle_world)

		robot_poses[i] = pos[:-1]
		yaws[i] = ori[2]
	return robot_poses, yaws

def get_robVels():
	global NAMESPACES
	global robots

	dyaws = np.zeros(len(NAMESPACES))
	for i, name in enumerate(NAMESPACES):

		linearVelocity, angularVelocity = sim.getObjectVelocity(robots[name])

		dyaws[i] = angularVelocity[2]
	return dyaws


def compute_tangent_points(center: np.ndarray, points: np.ndarray, side_sign: np.ndarray, r = 0.3) -> np.ndarray:
	
	# 1. Vector from base points (x0, y0) to center (xg, yg)
	delta = center - points  # Shape (N, 2)

	# 2. Distance D from each point to the circle center
	D = np.hypot(delta[:, 0], delta[:, 1])  # Shape (N,)

	# 3. Distance L from (x0, y0) to the tangent point (x', y')
	L = np.sqrt(D**2 - r**2)

	# 4. Base angle alpha to the center, and offset angle phi for the tangent
	alpha = np.arctan2(delta[:, 1], delta[:, 0])  # Shape (N,)
	phi = np.arcsin(r / D)  # Shape (N,)

	# 5. Final direction angle theta
	theta = alpha + side_sign * phi

	# 6. Compute (x', y') coordinates
	x_prime = points[:, 0] + L * np.cos(theta)
	y_prime = points[:, 1] + L * np.sin(theta)

	return np.column_stack((x_prime, y_prime))
# ================================================================================
# FUNCTION: sysCall_cleanup
# ================================================================================
def sysCall_cleanup():
	for name in NAMESPACES:
		if name in motors:
			lmotor, rmotor = motors[name]
			sim.setJointTargetForce(lmotor, 0.0)
			sim.setJointTargetForce(rmotor, 0.0)