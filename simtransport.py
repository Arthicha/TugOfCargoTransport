#python

import math
import numpy as np
import csv, os, sys
import matplotlib.pyplot as plt

import pandas as pd
from copy import deepcopy

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FOLDER_PATH)

from network.numpyrbf import RBFNetwork, load_model

# Python handles for CoppeliaSim (assumes `sim` is imported or provided in environment)
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
DATA_COLLECT = 1
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
	]
)


PHASE_TO_WIN = - np.pi / 2.0 if DIRECTION > 0 else  np.pi / 2.0
O_CENTRALIZE = np.array(
	[
		[0.0, 0.0, -PHASE_TO_WIN, -PHASE_TO_WIN],  # Robot 0 sending to others
		[0.0, 0.0, -PHASE_TO_WIN, -PHASE_TO_WIN],  # Robot 1 sending to others
		[PHASE_TO_WIN, PHASE_TO_WIN, 0.0, 0.0],  # Robot 2 sending to others
		[PHASE_TO_WIN, PHASE_TO_WIN, 0.0, 0.0],  # Robot 3 sending to others
	]
)

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

# ================================================================================
# FUNCTION: sysCall_init
# ================================================================================
def sysCall_init():
	global phis,  currents_offset
	global cargo, goal, rob_dummies, car_dummies
	global graph_handle, phi_stream_handles  # Add graph variables

	# Retrieve motor and robot handles
	for i, name in enumerate(NAMESPACES):
		lmotor = sim.getObject(f"/{name}/lmotor")
		rmotor = sim.getObject(f"/{name}/rmotor")

		motors[name] = (lmotor, rmotor)
		robots[name] = sim.getObject(f"/{name}")
		rob_dummies[name] = sim.getObject(f"/{name}/robo")
		car_dummies[name] = sim.getObject(f"/cargo_"+str(i+1))

		# Set baseline target velocity (0.2 rad/s)
		sim.setJointTargetVelocity(lmotor, 0.2)
		sim.setJointTargetVelocity(rmotor, 0.2)

	# Randomize initial phase angles in range [0, 2 * PI)
	phis = np.random.uniform(0.0, 2.0 * np.pi, size=4)
	graph_handle = sim.getObject("/Graph")

	# Set up graph streams for each element in phis (assuming 4 elements)
	phi_stream_handles = []
	colors = [
		[0, 0, 1],
		[0, 0.5, 1],
		[1, 0, 0],
		[1, 0.5, 0],
	]  # Red, Green, Blue, Yellow

	for i in range(len(NAMESPACES)): # timeseries graph
		stream_id = sim.addGraphStream(graph_handle,f"T{i}","Nm",0,colors[i % len(colors)])
		phi_stream_handles.append(int(stream_id))

		
	cargo = sim.getObject(f"/cargo")
	goal = sim.getObject(f"/goal")


	if (SCHEME == 'decentralized'):
		training_data = pd.read_csv(os.path.join(FOLDER_PATH,"checkpoints/"+("blue" if DIRECTION >= 1 else "red")+"win_nofilter.csv")).to_numpy()


		# 1. Initialize a 2x2 grid of subplots
		if PLOT:
			fig, axes = plt.subplots(2, 2, figsize=(12, 8))
			axes = axes.flatten()  # Flatten into 1D array of shape (4,) for easier indexing
	
		for i in range(len(NAMESPACES)):

			training_x = np.concatenate([np.sin(training_data[:,[i]]),np.cos(training_data[:,[i]])],axis=-1)
			training_y = training_data[:,[4+i]]

			currents_offset[i] = np.min(training_y)
			training_y = np.clip(training_y - currents_offset[i],0,None)

			# 3. Initialize and fit RBF Network
			
			if LOAD:
				forward_models.append(load_model(os.path.join(FOLDER_PATH,"checkpoints/"+("blue" if DIRECTION >= 1 else "red")+'model_rob'+str(i+1)+'.npz')))
				#forward_models[-1].sigma = 0.8

			else:
				model = RBFNetwork(n_centers=4, sigma=0.5,lr_centers=0.1,lr_weights=0.1)
				
				print(training_x.shape)
				print(training_y.shape)
				model.fit(training_x, training_y,verbose=0,epochs=5000)
				forward_models.append(model)

				model.save(os.path.join(FOLDER_PATH,"checkpoints/"+("blue" if DIRECTION >= 1 else "red")+'model_rob'+str(i+1)+'.npz'))
				
			predictions = forward_models[i].predict(training_x)
			grad_phi = forward_models[i].backward_phi(training_data[:,[i]], 1)
			tegotae_min[i] = np.min(grad_phi)
			tegotae_max[i] = np.max(grad_phi)

			# 4. Plot onto the current subplot (axis i)
			if PLOT:
				ax = axes[i]
				ax.plot(predictions, label='Predictions', linestyle='--')
				ax.plot(training_y, label='True Y', alpha=0.7)
				ax.set_title(f"Namespace: {NAMESPACES[i]}")
				ax.legend()

	# 5. Adjust spacing and display all 4 plots together
	if PLOT:
		plt.tight_layout()
		plt.show()


# ================================================================================
# FUNCTION: sysCall_actuation
# ================================================================================
def sysCall_actuation():
	global phis, tegotae, arrayhistory, goal, rob_dummies, car_dummies
	global yaws
	global tegotae_windup
	global graph_handle, phi_stream_handles

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
	# Step 1: Advance natural phase baseline and compute instantaneous force
	# ----------------------------------------------------------------------------
	phis = (phis + 1.0 * dt) % (2.0 * np.pi)
	force = np.clip(np.sin(phis), 0.0, 1.0)

	# ----------------------------------------------------------------------------
	# Step 2: Compute Kuramoto Coupling Feedback (Tegotae Matrix)
	# ----------------------------------------------------------------------------
	if SCHEME == 'centralized':
		delta_phi = (phis[:, None] + O_CENTRALIZE) - phis[None, :]
		tegotae = np.sum(K_CENTRALIZE * np.sin(delta_phi), axis=0)


	# ----------------------------------------------------------------------------
	# Step 2: Compute Tegotae Coupling Feedback (backpropagation)
	# ----------------------------------------------------------------------------
	if SCHEME == 'decentralized':
		for i, name in enumerate(NAMESPACES):
			forward_models[i].sigma = 0.8
			grad_phi_coarse = forward_models[i].backward_phi(phis[i], np.clip(currents[i]-currents_offset[i],0,None))
			forward_models[i].sigma = 0.5
			grad_phi_fine = forward_models[i].backward_phi(phis[i], np.clip(currents[i]-currents_offset[i],0,None))
			
			grad_phi = 0.2*grad_phi_coarse + 0.8*grad_phi_fine
			tegotae[i] = (grad_phi)/(tegotae_max[i] - tegotae_min[i])

		
		raw_drive = ETA * tegotae
		total_drive = raw_drive + tegotae_windup
		tegotae = np.clip(total_drive, -TEGOTAE_CLIP, TEGOTAE_CLIP)
		tegotae_windup += raw_drive - tegotae
	
	# ----------------------------------------------------------------------------
	# Step 3: Update phase angles using Kuramoto feedback
	# ----------------------------------------------------------------------------
	phis = (phis + dt*tegotae) % (2.0 * np.pi)
	
	# ----------------------------------------------------------------------------
	# Step 4: Compute turning adjustments & apply motor forces (Vectorized)
	# ----------------------------------------------------------------------------
	yaws = np.array(yaws)
	robotwise_goal = compute_tangent_points(goal_pose[:-1],robot_poses,np.array([+1,-1,-1,+1])*np.pi/2)
	distance_to_goal = robotwise_goal - robot_poses
	theta = np.arctan2(distance_to_goal[:,1],distance_to_goal[:,0])
	if DIRECTION > 0:
		theta[len(NAMESPACES)//2:] += np.pi 
	elif DIRECTION < 0:
		theta[:len(NAMESPACES)//2] += np.pi 

	turning = theta-yaws

	turning = -TURNING_GAINS * np.clip(np.sin(turning)*3 ,-1,1)  
	turning += TURNING_DGAINS*robot_dyaws*((turning+TURNING_DGAINS*robot_dyaws)*turning > 0)
	print(turning/TURNING_GAINS)

	left_forces = 0.5*force * np.clip(1.0 + force*turning,0,None)
	right_forces = 0.5*force * np.clip(1.0 - force*turning,0,None)

	# Apply target maximum forces (torque) to left and right motor joints
	for idx, name in enumerate(NAMESPACES):
		drive(name, float(left_forces[idx]),float(right_forces[idx]))
		

	# ----------------------------------------------------------------------------
	# Step 5: write
	# ----------------------------------------------------------------------------
	if (DATA_COLLECT):

		hist = []
		for data in [cargo_pose,cargo_orien,phis,currents,robot_poses,yaws]: #[phis,currents]
			if isinstance(data, np.ndarray):
				hist += data.flatten().tolist()
			elif isinstance(data, list):
				hist += np.array(data).flatten().tolist()

		arrayhistory.append(hist)

		if (len(arrayhistory) > (N_HISTORY)): 

			csv_path = os.path.join(FOLDER_PATH, "data/"+'data.csv')
			with open(csv_path, mode='a', newline='') as csv_file:
				writer = csv.writer(csv_file)
				writer.writerows(arrayhistory)
		
			arrayhistory = []


	for i, stream_id in enumerate(phi_stream_handles):
		sim.setGraphStreamValue(graph_handle, stream_id, force[i])

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