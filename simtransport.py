#python

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
# CONSTANTS & CONFIGURATION
# ================================================================================
SCHEME = "kuramoto" # "kuramoto" vs "adversarial cooperation"
DIRECTION = 1
ETA = 500
OMEGA = 1
DATA_COLLECT = 0
PLOT = 0
LOAD = 1

NAMESPACES = ["rob1", "rob2", "rob3", "rob4"]
TURNING_GAINS = np.array([1.0, 1.0, 1.0, 1.0])*0.2
TURNING_DGAINS = 0.7
TEGOTAE_CLIP = 0.5
TARGET_HZ = 20
DT_TARGET = 1.0 / TARGET_HZ
N_HISTORY = 200
ACCEPTANCE_RADIUS = 1.5

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
ti = 0
goalstages = 0
goallist = []
forward_models = []
phis = np.zeros(len(NAMESPACES))
tegotae = np.zeros(len(NAMESPACES))
tegotae_min = np.zeros(len(NAMESPACES))+0.0
tegotae_max = np.zeros(len(NAMESPACES))+0.0
tegotae_mean = np.zeros(len(NAMESPACES))
tegotae_windup = np.zeros(len(NAMESPACES))
yaws = np.zeros((len(NAMESPACES),))
positive_gain = np.zeros((len(NAMESPACES),))
negative_gain = np.zeros((len(NAMESPACES),))
current_history = np.zeros((400,len(NAMESPACES)))
currents_offset = np.zeros(len(NAMESPACES))
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
	global cargo, goal, rob_dummies, car_dummies, goallist
	global graph_handle, phi_stream_handles
	global positive_gain, negative_gain, forward_models

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
	subgoal1 = sim.getObject(f"/subgoal1")
	subgoal2 = sim.getObject(f"/subgoal2")
	goallist = [subgoal1,subgoal2,goal]

	# Randomize initial phase angles in range [0, 2 * PI)
	#phis = np.random.uniform(0.0, 2.0 * np.pi, size=4)
	phis = np.array([-np.pi/2,np.pi,0,np.pi/2])

	if SCHEME == "adversarial cooperation":
		color_prefix = "blue" if DIRECTION >= 1 else "red"
		data_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}win_nofilter.csv")
		training_data = pd.read_csv(data_path).to_numpy()
		
		if LOAD: # Route to dedicated loading or training function
			load_rbf_models(training_data)
		else:
			train_rbf_models(training_data)

		# find the positive/negative normalization gains
		for i in range(len(NAMESPACES)):
			grad=forward_models[i].backward_phi(training_data[:,[i]],training_data[:,[i+4]])
			positive_gain[i] = (grad[grad > 0].mean())
			negative_gain[i] = np.abs(grad[grad < 0].mean())

		
		if PLOT: # Plot predictions if enabled
			
			plot_training_results(training_data)


	# preparing graph
	graph_handle = sim.getObject("/Graph")

	colors = [
		[0.5, 0.5, 1],
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
	global ti, phis, tegotae, arrayhistory, goal, rob_dummies, car_dummies, goallist, goalstages
	global yaws, tegotae_windup, graph_handle, phi_stream_handles

	dt = DT_TARGET

	goalobj = goallist[goalstages]

	# ----------------------------------------------------------------------------
	# Step 0: sensing
	# ----------------------------------------------------------------------------
	cargo_pose = sim.getObjectPosition(cargo, sim.handle_world)
	goal_pose = sim.getObjectPosition(goalobj, sim.handle_world)
	cargo_orien = sim.getObjectOrientation(cargo, sim.handle_world)
	currents = get_estForce()
	robot_poses, yaws = get_robPoses()
	robot_dyaws = get_robVels()

	distance_to_goal = np.linalg.norm(np.array(goal_pose[:-1]) - np.array(cargo_pose[:-1]),axis=-1)
	print("Distance to goal "+str(goalstages)+" is "+ str(distance_to_goal))

	if (distance_to_goal < ACCEPTANCE_RADIUS) and (goalobj != goal):
		goalstages += 1

	# ----------------------------------------------------------------------------
	# Step 1: Advance natural phase baseline 
	# ----------------------------------------------------------------------------
	phis = (phis + OMEGA* dt) % (2.0 * np.pi)

	# ----------------------------------------------------------------------------
	# Step 2: Compute Coupling Feedback
	# ----------------------------------------------------------------------------
	if SCHEME == "kuramoto":
		tegotae = compute_centralized_coupling()
	elif SCHEME == "adversarial cooperation":
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
	#print(-np.sign(turning))
	left_forces = 2*0.5*force * np.clip(1.0 + force*turning,-1,1) 
	right_forces = 2*0.5*force * np.clip(1.0 - force*turning,-1,1) 
	
	#left_forces += np.clip(turning,-0.05,0.0)*(force == 0)
	#right_forces += np.clip(-turning,-0.05,0.0)*(force == 0)

	# Apply target maximum forces (torque) to left and right motor joints
	for idx, name in enumerate(NAMESPACES):
		drive(name, float(left_forces[idx]),float(right_forces[idx]), turning = force[idx]*turning[idx])

	# ----------------------------------------------------------------------------
	# Step 5: Recording & Visualization
	# ----------------------------------------------------------------------------
	if DATA_COLLECT:
		data = np.concatenate([np.array([ti]),cargo_pose,cargo_orien,phis.flatten(),currents.flatten(),robot_poses.flatten(),yaws.flatten()])
		#data = np.concatenate([phis.flatten(),currents.flatten()])
		record(data, skip=10)

	for i, stream_id in enumerate(phi_stream_handles):
		sim.setGraphStreamValue(graph_handle, stream_id, force[i])

	ti += 1.0/TARGET_HZ

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

	swap_indices = [1,0,3,2]

	#currents_offset = 0.998*currents_offset-0.002*currents

	tegotae_raw = np.zeros(len(NAMESPACES))
	for i in range(len(NAMESPACES)):
		adjusted_current = np.clip(currents[i],0,None)#np.clip(currents[i] - currents_offset[i], 0, None)

		for j, gamma in zip([i],[1]):
			#forward_models[j].sigma = 2.0
			#grad_phi_coarse = forward_models[j].backward_phi(phis[i], adjusted_current)
			#grad_phi_coarse[grad_phi_coarse > 0] *= 0.003/positive_gain[j]
			#grad_phi_coarse[grad_phi_coarse < 0] *= 0.003/negative_gain[j]

			#forward_models[j].sigma = 1.0
			grad_phi_fine = forward_models[j].backward_phi(phis[i], adjusted_current)
			grad_phi_fine[grad_phi_fine > 0] *= 0.003/positive_gain[j]
			grad_phi_fine[grad_phi_fine < 0] *= 0.003/negative_gain[j]
			
			tegotae_max[i] = 0.995*tegotae_max[i] + 0.005*(grad_phi_fine)

			#grad_phi_fine[grad_phi_fine > 0] *= 0.01/np.clip(tegotae_max[i],0.01,None)
			#grad_phi_fine[grad_phi_fine < 0] *= 0.01/np.clip(tegotae_min[i],0.01,None)
			'''if np.abs(grad_phi_fine) < 0.1*tegotae_max[i]:
				grad_phi_fine *= 0.0
			else:
				grad_phi_fine -= 0.1*tegotae_max[i]*np.sign(grad_phi_fine)'''

			grad_phi_fine -= 0.2*tegotae_max[i]#*np.sign(grad_phi_fine)

			grad_phi = gamma*grad_phi_fine#*1e-4/(1e-10+tegotae_max[i]) #0.1 * grad_phi_coarse + 0.9 * grad_phi_fine
			tegotae_raw[i] += grad_phi #/ (tegotae_max[i] - tegotae_min[i])
	
	#tegotae_raw[2:] *= -1

	if ti < 15:
		tegotae_raw*= 0.0
	#tegotae_raw[2:] *= -1
	raw_drive = ETA * tegotae_raw
	total_drive = raw_drive + tegotae_windup
	tegotae_clipped = np.clip(total_drive, -TEGOTAE_CLIP, TEGOTAE_CLIP)
	
	# Update anti-windup accumulator
	tegotae_windup += (raw_drive - tegotae_clipped)
	return tegotae_clipped



def compute_turning(goal_pose, robot_poses, yaws, robot_dyaws, force,offset = True):
	global TURNING_DGAINS, TURNING_GAINS

	"""Computes target heading alignment, turning adjustments, and wheel drive forces."""
	yaws_arr = np.array(yaws)
	
	if offset:
		robotwise_goal = compute_tangent_points(goal_pose[:-1],robot_poses,np.array([+1, -1, +1, -1]))
	else:
		robotwise_goal = np.expand_dims(goal_pose[:-1],axis=0)

	distance_to_goal = robotwise_goal - robot_poses
	theta = np.arctan2(distance_to_goal[:, 1], distance_to_goal[:, 0])



	if DIRECTION > 0:
		theta[len(NAMESPACES) // 2:] += np.pi
	elif DIRECTION < 0:
		theta[:len(NAMESPACES) // 2] += np.pi

	turning = np.sin(theta)*np.cos(yaws_arr) - np.cos(theta)*np.sin(yaws_arr)
	turning = -TURNING_GAINS * np.clip(np.sin(turning) * 3, -1, 1)
	turning += (
		TURNING_DGAINS
		* robot_dyaws
		* ((turning + TURNING_DGAINS * robot_dyaws) * turning > 0)
	)

	return turning

def record(list_of_data, skip=1):
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

		arrayhistory = arrayhistory[::skip]
		csv_path = os.path.join(FOLDER_PATH, "data/data.csv")
		with open(csv_path, mode="a", newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(arrayhistory)

		arrayhistory = []

def train_rbf_models(training_data):
	"""Trains RBF models from scratch, saves them, and sets gradient bounds."""
	n = len(NAMESPACES)
	swap_indices = [1,0,3,2]
	for i in range(n):
		training_x = np.concatenate([np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])],axis=-1)
		training_x_ = np.concatenate([np.sin(training_data[:, [swap_indices[i]]]), np.cos(training_data[:, [swap_indices[i]]])],axis=-1)
		training_y = training_data[:, [4 + i]] #- np.min(training_data[:, [4 + i]])
		#currents_offset[i] = np.min(training_data[:, [4 + i]])
		padded_y = np.pad(training_y, pad_width=((400 - 1,0),(0,0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400,axis=0)
		training_y = training_y - np.mean(windows_np,axis=-1)

		training_y_ = training_data[:, [4 + swap_indices[i]]] #- np.min(training_data[:, [4 + swap_indices[i]]])
		padded_y = np.pad(training_y_, pad_width=((400 - 1,0),(0,0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400,axis=0)
		training_y_ = training_y_ - np.mean(windows_np,axis=-1)

		training_x = np.concatenate([training_x,training_x_],axis=0)
		training_y = np.concatenate([training_y,training_y_],axis=0)
		
		#training_y = np.clip(training_y, 0, None)

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		model_path = os.path.join(FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{i + 1}.npz")

		model = RBFNetwork(n_centers=10, sigma=1.5, lr_centers=0.1, lr_weights=0.1)
		print(f"Training Robot {i + 1} | X: {training_x.shape}, Y: {training_y.shape}")
		model.fit(training_x, training_y, verbose=0, epochs=3000)
		model.save(model_path)

		forward_models.append(model)

		# Compute gradient bounds for Tegotae normalization
		#grad_phi = model.backward_phi(training_data[:, [i]], 1)
		#tegotae_min[i] = np.min(grad_phi)
		#tegotae_max[i] = np.max(grad_phi)


def load_rbf_models(training_data):
	"""Loads pre-trained RBF models and computes gradient bounds."""
	for i in range(len(NAMESPACES)):
		training_y = training_data[:, [4 + i]]
		currents_offset[i] = np.min(training_y)
		#training_y -= currents_offset[i]

		color_prefix = "blue" if DIRECTION >= 1 else "red"
		model_path = os.path.join(
			FOLDER_PATH, f"checkpoints/{color_prefix}model_rob{i + 1}.npz"
		)

		model = load_model(model_path)
		forward_models.append(model)

		# Compute gradient bounds for Tegotae normalization
		#grad_phi = model.backward_phi(training_data[:, [i]], 1)
		#tegotae_min[i] = np.min(grad_phi)
		#tegotae_max[i] = np.max(grad_phi)

		



def plot_training_results(training_data):
	"""Plots predictions vs true values across subplots for each robot."""
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	axes = axes.flatten()
	swap_indices = [2,3,0,1]
	for i in range(len(NAMESPACES)):
		training_x = np.concatenate([np.sin(training_data[:, [i]]), np.cos(training_data[:, [i]])],axis=-1,)
		training_y = training_data[:, [4 + i]]
		padded_y = np.pad(training_y, pad_width=((400 - 1,0),(0,0)), mode='edge')
		windows_np = sliding_window_view(padded_y, window_shape=400,axis=0)
		training_y = training_y - np.mean(windows_np,axis=-1)

		#forward_models[i].sigma = 2
		predictions = forward_models[i].predict(training_x)
		#forward_models[swap_indices[i]].sigma = 2
		predictions_ = forward_models[swap_indices[i]].predict(training_x)

		ax = axes[i]

		ax.plot(predictions_[:], label="Predictions_", linestyle="-",c='tab:red')
		ax.plot(predictions[:], label="Predictions", linestyle="-",c='tab:orange')
		ax.plot(training_y[:], label="True Y", alpha=0.7, linestyle="--",c='k')


		ax.set_title(f"Namespace: {NAMESPACES[i]}")
		ax.legend()

	plt.tight_layout()
	plt.show()

def drive(name, left_forces, right_forces, turning = 0):
	global motors

	lmotor, rmotor = motors[name]

	sim.setJointTargetForce(lmotor, left_forces)
	sim.setJointTargetForce(rmotor, right_forces)

	turninggain = 0.5
	vl = 1 + np.clip(2*turninggain*turning,-2,2)
	vr = 1 + np.clip(-2*turninggain*turning,-2,2)
	

	sim.setJointTargetVelocity(lmotor, 2*vl)
	sim.setJointTargetVelocity(rmotor, 2*vr)



def get_estForce():
	global NAMESPACES
	global rob_dummies, car_dummies

	currents = np.zeros(len(NAMESPACES))

	current_history[:-1] = current_history[1:]
	for i, name in enumerate(NAMESPACES):
		dumA = np.array(sim.getObjectPosition(rob_dummies[name], sim.handle_world))
		dumB = np.array(sim.getObjectPosition(car_dummies[name], sim.handle_world))

		if np.abs(current_history[:,i]).sum() == 0:
			current_history[:,i] = current_history[:,i]*0.00+np.linalg.norm(dumA-dumB,axis=-1)
		else:
			current_history[-1,i] =  np.linalg.norm(dumA-dumB,axis=-1)

		currents[i] = np.clip(np.linalg.norm(dumA-dumB,axis=-1) - np.percentile(current_history[:,i],50),0,None)
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


def compute_tangent_points(center: np.ndarray, points: np.ndarray, side_sign: np.ndarray, r = 0.7) -> np.ndarray:
	
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