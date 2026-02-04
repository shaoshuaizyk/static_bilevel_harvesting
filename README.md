# static_bilevel_harvesting
Use bi-level method to plan fruits harvesting

## Basic Fruit Harvesting Cycling Operation
1. Cartesian Arms, x to the right, y to the forward, z to the up. Arm indices start from bottom to up by 0, 1, 2... Each arms moves freely within a rectangle cell, with non-crossing constraints in z direction.
2. More specifically, when targeting a fruit, an arm moves laterally in the x-z plane, with its y-dof retracted, to avoid interfering with the canopy. Once it reaches the fruit's x-z world coordinates, it extends toward the fruit, detaches it (detachment time is $T_d$), retracts and drops the fruit on a conveyor.
3. Need to determine the harvesting sequences for arms, the arriving times (schedule) and the trajectories to minimize the finishing time (makespan).

## Bi-level Method
It has some problem now, need to improve.

## Zone-splitting + Trajectory Planning
Divid the fruits into zones vertically with roughly equal number of fruits. Each arm harvests one zone from left to right, according to the order of the arm. Then, do the trajectory planning based on the harvesting sequence and the fruit picking time constraint.

## Run test
Run the code in `main.ipynb` with jupyter notebook kernal or run `python static_bilevel.py` in the command line to see the results. Play around with the number of fruits, the number of arms, the random seeds to generate fruits, the gap and the draw trajectories function to see the results. Non-zero safety distance may not work in some extreme cases.