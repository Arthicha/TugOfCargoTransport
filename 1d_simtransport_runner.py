import os
import time
import json
import subprocess
import signal
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Get the directory of the current file as a string
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths using string concatenation
COPPELIASIM_EXE = r"C://Program Files//CoppeliaRobotics//CoppeliaSimEdu//coppeliaSim.exe"
SCENE_PATH = BASE_DIR + "//simulation//demo0_tug_of_war.ttt"
SIMULATION_SCRIPT_PATH = BASE_DIR + "//1d_simtransport.py"

# Hyperparameter grid
OMEGA_LIST = [1.0, 2.0]
OMGSCALE_LIST = [1.0/5, 1.0/3, 1.0, 3.0, 5.0]

BASE_PARAMS = {
	"NPOSITIVE": 2,
	"NNEGATIVE": 2,
	"SCHEME": "kuramoto",
	"DIRECTION": 1,
	"ETA": 500,
	"LOAD": 1,
	"DATA_COLLECT": 1,
	"SCRIPT_PATH": SIMULATION_SCRIPT_PATH
}

NUM_WORKERS = 1       # Number of parallel CoppeliaSim instances
BASE_PORT = 23000     # Starting ZeroMQ port


def execute_task(port_queue, task_id, omega, omgscale, duration_sec=30.0):
	"""Executes a simulation task by fetching an available port from the queue."""
	port = port_queue.get()
	
	try:
		print(f"[Task {task_id}] Connected to port {port} | OMEGA={omega}, OMGSCALE={omgscale:.3f}")
		
		client = RemoteAPIClient(host='127.0.0.1', port=port)
		sim = client.getObject('sim')
		params = BASE_PARAMS.copy()
		params["OMEGA"] = omega
		params["OMGSCALE"] = omgscale
		params["TASK_ID"] = task_id
		params["PORT"] = port



		json_params = json.dumps(params)
		sim.setStringSignal("ExperimentParams", json_params)

		sim.startSimulation()

		try:
			while True:
				t = sim.getSimulationTime()
				if t >= duration_sec:
					break
				time.sleep(0.1)
		except KeyboardInterrupt:
			# Prevent child worker from swallowing SIGINT silently
			raise
		finally:
			try:
				sim.stopSimulation()
				while sim.getSimulationState() != sim.simulation_stopped:
					time.sleep(0.1)
				sim.clearStringSignal("ExperimentParams")
			except Exception:
				pass  # Ignore API errors if terminating

		print(f"[Task {task_id}] Finished on port {port}")
		return f"Done Task {task_id}: OMEGA={omega}, OMGSCALE={omgscale}"

	finally:
		port_queue.put(port)


def launch_coppelia_instance(port):
	"""Launches a headless CoppeliaSim instance configured on a specific port."""
	cmd = [
		COPPELIASIM_EXE,
		"-h",                                          # Headless mode
		f"-GzmqRemoteApi.rpcPort={port}",             # Matches sim.getStringNamedParam('zmqRemoteApi.rpcPort')
		f"-GzmqRemoteApi.cntPort={port + 1}",         # Explicit continuous port assignment
		SCENE_PATH
	]
	return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():


	tasks = []
	task_counter = 1
	for i in range(4):
		for omegab in [1.0,0.8,0.5]:
			for omegar in [1.0,0.8,0.5]:
				omega = omegab
				omgscale = omegar/omegab
				tasks.append((task_counter, omega, omgscale))
				task_counter += 1

	total_tasks = len(tasks)
	ports = [BASE_PORT + i for i in range(NUM_WORKERS)]

	print(f"Total tasks: {total_tasks} | Parallel instances: {NUM_WORKERS}")

	manager = Manager()
	port_queue = manager.Queue()
	for p in ports:
		port_queue.put(p)

	processes = []
	print("\nLaunching headless CoppeliaSim background processes...")
	for port in ports:
		proc = launch_coppelia_instance(port)
		processes.append(proc)
		print(f" -> Started instance on port {port} (PID: {proc.pid})")

	time.sleep(5.0)

	executor = ProcessPoolExecutor(max_workers=NUM_WORKERS)

	try:
		futures = [
			executor.submit(execute_task, port_queue, t_id, omega, omgscale, 2*60.0)
			for t_id, omega, omgscale in tasks
		]

		for future in as_completed(futures):
			print(f"[Result] {future.result()}")

	except KeyboardInterrupt:
		print("\n[Ctrl+C detected] Force shutting down all processes...")
		# Force terminate executor processes
		executor.shutdown(wait=False, cancel_futures=True)
		
	finally:
		# Cleanup background CoppeliaSim instances
		print("\nTerminating background CoppeliaSim processes...")
		for proc in processes:
			try:
				proc.terminate()
				proc.wait(timeout=2)
			except Exception:
				proc.kill()  # Hard kill on timeout or error
		
		# Shutdown multiprocessing manager cleanly
		manager.shutdown()
		print("All processes closed successfully.")


if __name__ == "__main__":
	main()