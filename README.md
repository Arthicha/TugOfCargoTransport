# Tug of War: How Adversarial Robots Lead to Emergence Cooperation

![TugOfRobots](figures/cargo_transport.png)

## Requirements (tested platform)

- Python 3.12
- python standard modules (numpy, matplotlib, etc)
- CoppeliaSim V4.4 [download](https://www.coppeliarobotics.com/)

## Setting Up

- To run a python script in CoppeliaSim, navigate to `%APPDATA%/CoppeliaSim` and add 
`defaultPython = "<path to python>"`, e.g., "C:\Users\myuser\AppData\Local\Programs\Python\Python312\python.exe", to `usrset.txt`.

## Running the Experiment Yourself

1. Open Coppelia scene `tortel_cargoexp.ttt`

2. In cargo's child scrpt, set EXTERNAL_FILE_PATH to "simtransport.py". For example, "D:\myname\projectname\simtransport.py"

3. Run (>) in the CoppeliaSim. CoppeliaSim will automatically run simtransport.py for you.

4. Enjoy!

## Pipeline (AI-summary)

![Pipeline](figures/oversimplified_pipeline.png)

In phase 1 (Data Collection): set SCHEME to "centralized" and DATA_COLLECT to "True". 

In phase 2 (Pre-training): set SCHEME to "decentralized", DATA_COLLECT to "False", PLOT to "True", and LOAD to "False". Also, rename the data file to "bluewin_nofilter.csv" (DIRECTION = 1) or "redwin_nofilter.csv" (DIRECTION = -1). Make sure the model fit to the data well, as shown below:

![goodfit](figures/rbf_training.png)

In phase 3 (Decentralized Deployment): set SCHEME to "decentralized", DATA_COLLECT to "False", PLOT to "True" or "False", and LOAD to "True" to bypass the training process. Here, you may need to tune the learning/adaptation/coupling rate ETA.

## Code Structure (AI-summary)

![Structure](figures/oversimplified_diagram.png)


## Demo1: Toward a Goal

![DEMO1](figures/demo1_toward_a_goal.png)


## Configure the Program

In "simtransport.py", there are 2 configuration argument you can play. 

<item> SCHEME: control scheme, either "centralized" (standart kuramoto coupling enforcing phase offset) vs "decentralized" (implecit coupling via force feedback).

<item> DIRECTION: target moving direction. 1 toward the blue team, -1 toward the red team.

For example, when (SCEME,DIRECTION) = ("centralized",1) the robot collective will moves toward the right (blue team) using explicit kuramoto coupling. When (SCEME,DIRECTION) = ("decentralized",-1), the robot collective will move toward the left (red team) using implecit coupling without any explicit communication between robots.

## Contact

Arthicha Srisuchinnawong (zumoarthicha@gmail.com)