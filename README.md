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

2. In cargo's child scrpt, set EXTERNAL_FILE_PATH to "simtransport.py" or "multisimtransport.py". For example, "D:\myname\projectname\simtransport.py"

3. Run (>) in the CoppeliaSim. CoppeliaSim will automatically run simtransport.py for you.

4. Enjoy!

## Pipeline (AI-summary)

![Pipeline](figures/oversimplified_pipeline.png)

In phase 1 (Data Collection): set SCHEME to "kuramoto" and DATA_COLLECT to "True". 

In phase 2 (Pre-training): set SCHEME to "adversarial cooperation", DATA_COLLECT to "False", PLOT to "True", and LOAD to "False". Also, rename the data file to "bluewin_nofilter.csv" (DIRECTION = 1) or "redwin_nofilter.csv" (DIRECTION = -1). Make sure the model fit to the data well, as shown below:

![goodfit](figures/rbf_training.png)

In phase 3 (Deployment): set SCHEME to "adversarial cooperation", DATA_COLLECT to "False", PLOT to "True" or "False", and LOAD to "True" to bypass the training process. Here, you may need to tune the learning/adaptation/coupling rate ETA.

## Code Structure (AI-summary)

![Structure](figures/oversimplified_diagram.png)

## Demo0: 1D Tug-of-War Model

![DEMO0](figures/demo0_tugofwar_1d_model.png)

## Demo1: Toward a Goal

![DEMO1](figures/demo1_toward_a_goal.png)

## Demo 3: Candy Experiment (Ant)

![DEMO3](figures/demo3_candy.PNG)

## Demo 4: Maze Experiment (Ant)

![DEMO4](figures/demo4_maze.PNG)

## Demo 5: Traffic (Motor Protein)

![DEMO5](figures/demo5_protein.png)

## Configure the Program

In "simtransport.py", there are 2 configuration argument you can play. 

<item> SCHEME: control scheme, either "kuramoto" (standart kuramoto coupling enforcing phase offset) vs "adversarial cooperation" (implecit coupling via force feedback).

<item> DIRECTION: target moving direction. 1 toward the blue team, -1 toward the red team.

For example, when (SCEME,DIRECTION) = ("kuramoto",1) the robot collective will moves toward the right (blue team) using explicit kuramoto coupling. When (SCEME,DIRECTION) = ("adversarial cooperation",-1), the robot collective will move toward the left (red team) using implecit coupling without any explicit communication between robots.

For a newer version (multisimtransport.py), you can set the following parameter within coppeliasim's child script.
```
PARAMS = {
    "NPOSITIVE": 4,           # Number of positive team robots
    "NNEGATIVE": 4,           # Number of negative team robots
    "SCHEME": "adversarial_cooperation",  # "kuramoto" vs "adversarial_cooperation"
    "DIRECTION": 1,           # Motion direction flag
    "ETA": 500*1,               # Tegotae coupling gain
    "OMEGA": 1,               # Natural frequency
    "LOAD": 1,                # Load pre-trained models
    "DATA_COLLECT": 0,        # Enable/disable data logging
}
```

## Contact

Arthicha Srisuchinnawong (zumoarthicha@gmail.com)
Atanu Chatterjee (atanu.chatterjee@colorado.edu)