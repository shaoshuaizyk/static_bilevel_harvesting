import gurobipy as gp
from gurobipy import GRB, tuplelist
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from itertools import combinations


class StaticBilevelModel:
    def __init__(self, fruits, height, length, arms_n, vx_max, vy_max, vz_max, ax_max, ay_max, az_max, dt=0.1, safety_distance=0.15, initial_positions=None):
        """
        Docstring for __init__
        :param fruits: list of fruit 3D positions, as well as their detaching times
        :param height: height of the area
        :param length: length of the area
        :param arms_n: number of arms
        :param vx_max: maximum velocity in x direction
        :param vy_max: maximum velocity in y direction
        :param vz_max: maximum velocity in z direction
        :param ax_max: maximum acceleration in x direction
        :param ay_max: maximum acceleration in y direction
        :param az_max: maximum acceleration in z direction
        :param dt: time step for the simulation
        :param safety_distance: minimum safety distance between arms
        :param initial_positions: initial positions of the arms
        """
        fruits.sort(key=lambda fruit: fruit.coordinate.x)
        self.fruits = fruits
        self.height = height
        self.length = length
        self.arms_n = arms_n
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.vz_max = vz_max
        self.ax_max = ax_max
        self.ay_max = ay_max
        self.az_max = az_max
        self.dt = dt
        self.safety_distance = safety_distance
        self.initial_positions = initial_positions if initial_positions is not None else [Coordinate(length/2.0, 0.0, height/(arms_n+1)*(r+1)) for r in range(arms_n)]
        self.harvesting_times = [np.ceil((fruit.detach_time + self.traveling_time(fruit.coordinate.y, self.vy_max, self.ay_max)*2)/self.dt)*self.dt for fruit in self.fruits]

    def bilevel_trajectory_planning(self, guess_tau=100, MIPFocus=1, Heuristics=0.707, log_to_console=True, threads=-1, time_limit=3600):
        """
        Plan the trajectories for the arms to harvest the fruits.
        """
        with gp.Model("Bilevel_Trajectory_Planning") as bilevel_model:
            fruit_n = len(self.fruits)
            N = [*range(fruit_n+2)] # Indices of fruits and dummy start/end nodes
            R = [*range(self.arms_n)] # Indices of arms
            # Decision variables
            # print(f"Adding decision variables...")
            # tic = time.time()
            tau = bilevel_model.addVar(vtype=GRB.CONTINUOUS, name="makespan")  # Total finishing time
            E = bilevel_model.addVars(R, N, N, vtype=GRB.BINARY, name="edges")  # Edge selection variables
            V = bilevel_model.addVars(R, N, vtype=GRB.BINARY, name="vertices")  # Vertex assignment variables
            bilevel_model._E = E
            bilevel_model._V = V
            bilevel_model._tau = tau
            tau.start = guess_tau

            # toc = time.time()
            # print(f"Decision variables added in {toc - tic:.3f} seconds.")

            # Constraints and objective
            # print(f"Adding constraints...")
            # tic = time.time()
            # Constraints 0.1: self connection regarded as 0
            bilevel_model.addConstrs((E[r, i, i] == 0 for r in R for i in N), name="no_self_connection")
            # Constraints 0.2: Each arm harvesting number should not be too large or too small
            average_each_arm_harvest = round(fruit_n / self.arms_n)
            bilevel_model.addConstrs((gp.quicksum(V[r, i] for i in N[1:-1]) >= max(1, average_each_arm_harvest - 1) for r in R), name="each_arm_harvest_at_least_some_fruits")
            bilevel_model.addConstrs((gp.quicksum(V[r, i] for i in N[1:-1]) <= average_each_arm_harvest + 1 for r in R), name="each_arm_harvest_at_most_some_fruits")
            # Constraints 0.3: Never come back to dummy start, and start from dummy end
            bilevel_model.addConstrs((E[r, i, 0] == 0 for r in R for i in N), name="no_come_back_to_dummy_start")
            bilevel_model.addConstrs((E[r, N[-1], j] == 0 for r in R for j in N), name="no_start_from_dummy_end")
            # Constraints 0.4: No undirected edges
            bilevel_model.addConstrs((E[r, i, j] + E[r, j, i] <= 1 for r in R for i in N for j in N), name="no_undirected_edges")

            # Constraints 1: high-level route planning
            # Constraints 1.1: Harvest all fruits
            bilevel_model.addConstr((gp.quicksum(V[r, i] for r in R for i in N[1:-1]) == fruit_n), name="all_fruits_harvested")

            # Constraints 1.2: start from dummy start and end at dummy end
            bilevel_model.addConstrs((gp.quicksum(E[r, 0, j] for j in N) == 1 for r in R), name="start_from_dummy_start")
            bilevel_model.addConstrs((gp.quicksum(E[r, i, N[-1]] for i in N) == 1 for r in R), name="end_at_dummy_end")

            # Constraints 1.3: connectivity of each tour
            bilevel_model.addConstrs((gp.quicksum(E[r, i, k] for i in N[:-1]) == V[r, k] for r in R for k in N[1:]), name="connectivity_incoming")
            bilevel_model.addConstrs((gp.quicksum(E[r, k, j] for j in N[1:]) == V[r, k] for r in R for k in N[:-1]), name="connectivity_outgoing")

            # Constraints 1.4: visiting each fruit at most once
            bilevel_model.addConstrs((gp.quicksum(V[r, k] for r in R) <= 1 for k in N[1:-1]), name="visit_each_fruit_at_most_once")

            # Adding the objective
            bilevel_model.setObjective(tau, GRB.MINIMIZE)

            # toc = time.time()
            # print(f"Constraints added in {toc - tic:.3f} seconds.")

            # Solve the bilevel model
            # print("Optimizing the bilevel model...")
            # tic = time.time()
            # Set parameters
            bilevel_model.setParam('MIPFocus', MIPFocus)
            bilevel_model.setParam('Heuristics', Heuristics)
            bilevel_model.setParam('LogToConsole', log_to_console)
            if threads > 0:
                bilevel_model.setParam('Threads', threads)
            bilevel_model.setParam('TimeLimit', time_limit)
            if log_to_console:
                bilevel_model.setParam('LogToConsole', 1)
            else:
                bilevel_model.setParam('LogToConsole', 0)

            bilevel_model.setParam('lazyConstraints', 1)
            bilevel_model.optimize(lambda model, where: self.trajectory_plan_callback(model, where))
            # toc = time.time()
            # print(f"Bilevel model optimized in {toc - tic:.3f} seconds.")

            # Print the solution
            if bilevel_model.status == GRB.OPTIMAL or bilevel_model.status == GRB.INTERRUPTED or bilevel_model.status == GRB.TIME_LIMIT:
                print(f"Optimal makespan: {bilevel_model._tau.X:.3f} seconds")
                # Extract the solution
                E_sol = bilevel_model.getAttr('X', E)
                E_selected = tuplelist((r, i, j) for r, i, j in E.keys() if E_sol[r, i, j]> 0.5)
                sequences, _ = self.get_sequences_and_times(E_selected)
                makespan = bilevel_model.getAttr('X', tau)
                self.arms_plan = ArmsPlan(harvesting_sequences=sequences, finishing_time=makespan)
                assert self.trajectory_plan_on_sequences(self.arms_plan, alpha=0, use_makepan_as_constraint=True, log_to_console=True), "Final trajectory is infeasible!"
            else:
                print("No feasible solution found.")
                assert False, "Bilevel trajectory planning failed!"
        print(f"Bilevel trajectory planning completed.")

    def trajectory_plan_callback(self, model, where):
        """
        Callback function for the bilevel model to add lower-level trajectory planning constraints.
        """
        if where == GRB.Callback.MIPSOL:
            # Get the current solution
            cur_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            E = model._E
            E_sol = model.cbGetSolution(E)
            V = model._V
            V_sol = model.cbGetSolution(V)
            R = range(self.arms_n)
            N = range(len(self.fruits)+2)

            E_selected = tuplelist((r, i, j) for r, i, j in E.keys() if E_sol[r, i, j]> 0.5)

            # Eliminate the cycles
            cycles = self.get_cycles(E_selected)
            if cycles is not None:
                for r in R:
                    cur_cycles = cycles[r]
                    for cur_cycle in cur_cycles:
                        model.cbLazy(gp.quicksum(E[r, i, j] for i, j in combinations(cur_cycle, 2)) <= len(cur_cycle) - 2)
                return
            
            sequences, _ = self.get_sequences_and_times(E_selected)
            arms_plan = ArmsPlan(harvesting_sequences=sequences, finishing_time=cur_obj)

            is_feasible = self.trajectory_plan_on_sequences(arms_plan, alpha=1, log_to_console=False) # Update the arms plan and the current finishing time
            if is_feasible:
                print(f"Current sequences {sequences} is feasible with minimum finishing time {arms_plan.finishing_time:.3f}.")
                model.cbLazy(model._tau >= arms_plan.finishing_time)
            else:
                print(f"Current sequences {sequences} is INFEASIBLE!!!!!!! Adding lazy constraints to eliminate this solution.")
                model.cbLazy(gp.quicksum(E[r, i, j] for r, i, j in E_selected) <= len(E_selected) - 1)

    def trajectory_plan_on_sequences(self, arms_plan, alpha=1, use_makespan_as_constraint=False, log_to_console=True, heuristics=0.707, gap=1e-4, tolerance=1e-4):
        """
        Plan the robot arms trajectories based on the given harvesting sequences. The dummy start and end are not necesarry. Use alpha to balance the finishing time and the smoothness of the trajectories.
        alpha = 1: only minimize the finishing time
        alpha = 0: only minimize the average acceleration
        """
        assert arms_plan.harvesting_sequences is not None, "Harvesting sequences must be provided."
        if log_to_console:
            print(f"Planning trajectories for sequences {arms_plan.harvesting_sequences} with arriving times {arms_plan.arriving_times} and initial finishing time {arms_plan.finishing_time:.3f}...")

        is_feasible = True
        with gp.Model("Trajectory_Planning_On_Sequences") as traj_model_with_seq:
            harvesting_sequences = arms_plan.harvesting_sequences
            arriving_times = arms_plan.arriving_times # it could be None
            tau = arms_plan.finishing_time * (1 + gap) # A bit larger than the initial finishing time
            Tmax = tau + self.dt*2
            Tmax_steps = int(np.ceil(Tmax / self.dt))
            M = 10*Tmax  # A large constant for big-M method
            R = range(self.arms_n)
            Ts = [*range(Tmax_steps)]  # Time steps
            _seq = tuplelist([(r, k) for r in R for k in range(len(harvesting_sequences[r]))])
            _binaries = tuplelist([(r, k, ts) for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts])

            # Decision variables
            Schedule = traj_model_with_seq.addVars(_seq, vtype=GRB.INTEGER, lb=0, ub=Tmax_steps, name="Schedule_time_steps")  # Arriving time steps for each arms arriving at each fruit
            X_wpts = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=0, ub=self.length, name="X_waypoints")  # X waypoints for each arm at each time step
            Z_wpts = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=0, ub=self.height, name="Z_waypoints")  # Z waypoints for each arm at each time step
            acc_x = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=-self.ax_max, ub=self.ax_max, name="acceleration_x")  # X acceleration for each arm at each time step
            acc_z = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=-self.az_max, ub=self.az_max, name="acceleration_z")  # Z acceleration for each arm at each time step
            abs_acc_x = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=0, ub=self.ax_max, name="abs_acceleration_x")  # Absolute X acceleration for each arm at each time step
            abs_acc_z = traj_model_with_seq.addVars(R, Ts, vtype=GRB.CONTINUOUS, lb=0, ub=self.az_max, name="abs_acceleration_z")  # Absolute Z acceleration for each arm at each time step
            average_acc_x = traj_model_with_seq.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.ax_max, name="average_acceleration_x")  # Average absolute X acceleration for each arm
            average_acc_z = traj_model_with_seq.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.az_max, name="average_acceleration_z")  # Average absolute Z acceleration for each arm
            Bh1 = traj_model_with_seq.addVars(_binaries, vtype=GRB.BINARY, name="picking_start_timestep")  # Binary variable indicating if picking starts at time step ts for arm r at fruit k
            Bh2 = traj_model_with_seq.addVars(_binaries, vtype=GRB.BINARY, name="picking_end_timestep")  # Binary variable indicating if picking ends at time step ts for arm r at fruit k
            Bi = traj_model_with_seq.addVars(_binaries, vtype=GRB.BINARY, name="is_harvesting_at_timestep")  # Binary variable indicating if arm r is harvesting fruit k at time step ts
            makespan = traj_model_with_seq.addVar(vtype=GRB.CONTINUOUS, name="makespan")  # Finishing time

            # Constraints
            # Constraints 0: use makespan as constraint if needed
            if use_makespan_as_constraint:
                traj_model_with_seq.addConstr(makespan == tau, name="makespan_determination")
                if arriving_times is not None:
                    traj_model_with_seq.addConstrs((Schedule[r, k] == arriving_times[r][k] for r in R for k in range(len(harvesting_sequences[r]))), name="arriving_time_from_initial_solution")
            # Constraints 1: determine the harvesting seqeunces timeline.
            traj_model_with_seq.addConstrs((Schedule[r, k]*self.dt + self.harvesting_times[harvesting_sequences[r][k]] + self.travel_time_f2f(self.fruits[harvesting_sequences[r][k]], self.fruits[harvesting_sequences[r][k+1]]) <= Schedule[r, k+1]*self.dt for r in R for k in range(len(harvesting_sequences[r])-1)), name="harvesting_sequence_timeline")

            # Constraints 2: initial positions
            traj_model_with_seq.addConstrs((X_wpts[r, 0] == self.initial_positions[r].x for r in R), name="initial_x_position")
            traj_model_with_seq.addConstrs((Z_wpts[r, 0] == self.initial_positions[r].z for r in R), name="initial_z_position")
            traj_model_with_seq.addConstrs((Schedule[r, 0]*self.dt >= self.travel_time_f2f(self.initial_positions[r], self.fruits[harvesting_sequences[r][0]])) for r in R)

            # Constraints 3: kinematic constraints
            traj_model_with_seq.addConstrs((X_wpts[r, ts+1] - X_wpts[r, ts] >= self.dt*(-self.vx_max) for r in R for ts in Ts[:-1]), name="max_negative_x_velocity")
            traj_model_with_seq.addConstrs((X_wpts[r, ts+1] - X_wpts[r, ts] <= self.dt*(self.vx_max) for r in R for ts in Ts[:-1]), name="max_positive_x_velocity")
            traj_model_with_seq.addConstrs((Z_wpts[r, ts+1] - Z_wpts[r, ts] >= self.dt*(-self.vz_max) for r in R for ts in Ts[:-1]), name="max_negative_z_velocity")
            traj_model_with_seq.addConstrs((Z_wpts[r, ts+1] - Z_wpts[r, ts] <= self.dt*(self.vz_max) for r in R for ts in Ts[:-1]), name="max_positive_z_velocity")
            traj_model_with_seq.addConstrs((X_wpts[r, ts] - 2*X_wpts[r, ts+1] + X_wpts[r, ts+2] == self.dt**2 * acc_x[r, ts] for r in R for ts in Ts[:-2]), name="x_acceleration_definition")
            traj_model_with_seq.addConstrs((Z_wpts[r, ts] - 2*Z_wpts[r, ts+1] + Z_wpts[r, ts+2] == self.dt**2 * acc_z[r, ts] for r in R for ts in Ts[:-2]), name="z_acceleration_definition")
            # Absolute acceleration definition
            for r in R:
                for ts in Ts:
                    traj_model_with_seq.addConstr(abs_acc_x[r, ts] == gp.abs_(acc_x[r, ts]), name=f"absolute_x_acceleration_definition_r{r}_ts{ts}")
                    traj_model_with_seq.addConstr(abs_acc_z[r, ts] == gp.abs_(acc_z[r, ts]), name=f"absolute_z_acceleration_definition_r{r}_ts{ts}")
            # Average acceleration definition
            traj_model_with_seq.addConstr(average_acc_x == abs_acc_x.sum()/(len(R)*len(Ts)*self.dt), name="average_x_acceleration_definition")
            traj_model_with_seq.addConstr(average_acc_z == abs_acc_z.sum()/(len(R)*len(Ts)*self.dt), name="average_z_acceleration_definition")

            # Constraints 4: picking time constraints
            # Define the picking start time binaries
            # If ts >= Schedule[r, k], then Bh1[r, k, ts] = 1, otherwise Bh1[r, k, ts] = 0
            traj_model_with_seq.addConstrs((ts*self.dt >= Schedule[r, k]*self.dt - M*(1 - Bh1[r, k, ts]) for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="picking_start_time_definition_lb")
            traj_model_with_seq.addConstrs((ts*self.dt <= Schedule[r, k]*self.dt - tolerance + M*Bh1[r, k, ts] for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="picking_start_time_definition_ub")
            # Define the picking end time binaries
            # If ts <= Schedule[r, k] + harvesting_time, then Bh2[r, k, ts] = 1, otherwise Bh2[r, k, ts] = 0
            traj_model_with_seq.addConstrs((ts*self.dt <= Schedule[r, k]*self.dt + self.harvesting_times[harvesting_sequences[r][k]] + M*(1 - Bh2[r, k, ts]) for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="picking_end_time_definition_ub")
            traj_model_with_seq.addConstrs((ts*self.dt >= Schedule[r, k]*self.dt + self.harvesting_times[harvesting_sequences[r][k]] + tolerance - M*Bh2[r, k, ts] for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="picking_end_time_definition_lb")
            # Define the Bi is harvesting binaries. When the Bh1[ts] is True and Bh2[ts] is True, Bi is 1, otherwise, Bi is 0. In general, if a1 is True and a2 is True, then a3 is True, otherwise a3 is False. The equivalent expression is a3 >= a1 + a2 - 1, a3 <= a1 and a3 <= a2.
            traj_model_with_seq.addConstrs((Bi[r, k, ts] >= Bh1[r, k, ts] + Bh2[r, k, ts] - 1 for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="is_harvesting_definition_lb")
            traj_model_with_seq.addConstrs((Bi[r, k, ts] <= Bh1[r, k, ts] for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="is_harvesting_definition_ub1")
            traj_model_with_seq.addConstrs((Bi[r, k, ts] <= Bh2[r, k, ts] for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="is_harvesting_definition_ub2")
            # When arm r is harvesting fruit k at time step ts, the arm should be at the fruit position
            traj_model_with_seq.addConstrs(((Bi[r, k, ts] == 1) >> (X_wpts[r, ts] == self.fruits[harvesting_sequences[r][k]].coordinate.x) for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="x_position_during_harvesting")
            traj_model_with_seq.addConstrs(((Bi[r, k, ts] == 1) >> (Z_wpts[r, ts] == self.fruits[harvesting_sequences[r][k]].coordinate.z) for r in R for k in range(len(harvesting_sequences[r])) for ts in Ts), name="z_position_during_harvesting")

            # Constraints 5: safety distance constraints
            traj_model_with_seq.addConstrs((Z_wpts[r+1, ts] >= Z_wpts[r, ts] + self.safety_distance for r in R[:-1] for ts in Ts), name="safety_distance_between_arms")

            # Constraints 6: finishing time definition
            traj_model_with_seq.addConstrs((makespan >= Schedule[r, len(harvesting_sequences[r])-1]*self.dt + self.harvesting_times[harvesting_sequences[r][-1]] for r in R), name="finishing_time_definition")

            # Initial guess
            makespan.start = tau

            # Objective: minimize the finishing time and the average acceleration
            traj_model_with_seq.setObjective(alpha*makespan + (1-alpha)*(average_acc_x + average_acc_z), GRB.MINIMIZE)

            # Set parameters and optimize
            traj_model_with_seq.setParam('MIPFocus', 1)
            traj_model_with_seq.setParam('Heuristics', heuristics)
            traj_model_with_seq.setParam('LogToConsole', 1 if log_to_console else 0)
            traj_model_with_seq.setParam('OutputFlag', 1 if log_to_console else 0)
            traj_model_with_seq.setParam('MIPGap', gap)

            traj_model_with_seq.optimize()

            if traj_model_with_seq.status == GRB.OPTIMAL:
                schedule = traj_model_with_seq.getAttr('X', Schedule)
                arms_plan.arriving_times = [[schedule[r, k]*self.dt for k in range(len(harvesting_sequences[r]))] for r in R]
                finishing_time = makespan.X
                arms_plan.finishing_time = finishing_time
                max_ts = int(finishing_time / self.dt)
                xs = traj_model_with_seq.getAttr('X', X_wpts)
                zs = traj_model_with_seq.getAttr('X', Z_wpts)
                paths_x = np.array([[xs[r, ts] for ts in range(max_ts)] for r in R])
                paths_z = np.array([[zs[r, ts] for ts in range(max_ts)] for r in R])
                arms_plan.paths_x = paths_x
                arms_plan.paths_z = paths_z
                # if log_to_console:
                print(f"Trajectory planning successful with finishing time {finishing_time:.3f}.")
                is_feasible = True
            else:
                # if log_to_console:
                print(f"Trajectory planning failed. Model status: {traj_model_with_seq.status}.")
                is_feasible = False
        return is_feasible
        
    def get_cycles(self, edges):
        """
        Detect cycles in the selected edges.
        :param edges: selected edges from the bilevel model
        :return: cycles detected for each arm
        """
        cycles = [[] for _ in range(self.arms_n)]
        n = len(self.fruits) + 2
        has_cycle = False
        for r in range(self.arms_n):
            # Build path graph based on the edges. There might be cycles in the graph
            cur_G = [-1 for _ in range(n)] # -1 means initial value, -2 means visited, other values mean the neighbor node
            for _, i, j in edges.select(r, '*', '*'):
                assert cur_G[i] < 0 or cur_G[i] == j, f"Current node {i} has more than one neighbors {cur_G[i]} and {j}."
                cur_G[i] = j
            # print(f"Current arm {r+1} has edges {edges.select(r, '*', '*')} and cur_G {cur_G}.")
            # Find the cycles
            for i in range(n):
                cur_node = i
                if cur_G[cur_node] < 0: # Have been visited or not connected
                    continue
                cycle = [cur_node]
                while cur_G[cur_node] >= 0:
                    pre = cur_node
                    cur_node, cur_G[pre] = cur_G[cur_node], -2
                    cycle.append(cur_node)
                    # print(f"Cur node is {cur_node}. Current cycle is {cycle}. Cur G is {cur_G}.")
                if cur_G[cur_node] == -1:
                    # Current path has no cycles
                    continue
                if cur_G[cur_node] == -2:
                    # Current path has cycles
                    has_cycle = True
                    cycles[r].append(cycle)
                else:
                    raise Exception(f"Current path has cycles. Please check cur edges {edges.select(r, '*', '*')}.")
        return cycles if has_cycle else None
    
    def get_sequences_and_times(self, edges, solution_times=None):
        """
        Get the sequences and arriving times from the selected connected edges. The sequences should start from dummy start to dummy end.
        """
        # print(f"Edges are {edges}.")
        R = [*range(self.arms_n)]
        n = len(self.fruits) + 2
        sequences = [[] for r in R]
        updated = False
        if solution_times is not None:
            arriving_times = [[] for r in R]
        for r in R:
            # Build path graph based on the edges
            cur_G = [-1 for _ in range(n)]
            for _, i, j in edges.select(r, '*', '*'):
                assert cur_G[i] < 0 or cur_G[i] == j, f"Current node {i} has more than one neighbors {cur_G[i]} and {j}."
                cur_G[i] = j
            cur_node = 0
            count = 0
            while cur_node != n-1:
                neighbor = cur_G[cur_node]
                assert neighbor > 0, f"Current node {cur_node} has no neighbor in cur_G {cur_G}."
                cur_node = cur_G[cur_node]
                if cur_node < n-1:
                    updated = True
                    sequences[r].append(cur_node-1)
                    if solution_times is not None:
                        arriving_times[r].append(arriving_times[r, cur_node]*self.dt)
                    count += 1
                assert count < n-1, f"Count is {count}. Current path has cycles. Please check cur edges {edges.select(r, '*', '*')}."
        assert updated, f"No solution found! Cur edges {edges}."
        return (sequences, None if solution_times is None else arriving_times)

        
    def traveling_time(self, distance, v_max, a_max):
        """
        Calculate the traveling time for a given distance, maximum velocity, and maximum acceleration.
        :param distance: distance to travel
        :param v_max: maximum velocity
        :param a_max: maximum acceleration
        :return: traveling time
        """
        assert distance >= 0, "Distance must be non-negative"
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel**2

        if distance < 2 * d_accel:
            t_total = 2 * np.sqrt(distance / a_max)
        else:
            d_const = distance - 2 * d_accel
            t_const = d_const / v_max
            t_total = 2 * t_accel + t_const

        return np.ceil(t_total/self.dt) * self.dt  # Round up to nearest time step
    
    def travel_time_f2f(self, fruit1, fruit2):
        """
        Calculate the traveling time between two fruits.
        :param fruit1: first fruit or initial position
        :param fruit2: second fruit or initial position
        :return: traveling time
        """
        dx = abs(fruit1.coordinate.x if hasattr(fruit1, 'coordinate') else fruit1.x - fruit2.coordinate.x if hasattr(fruit2, 'coordinate') else fruit2.x)
        dz = abs(fruit1.coordinate.z if hasattr(fruit1, 'coordinate') else fruit1.z - fruit2.coordinate.z if hasattr(fruit2, 'coordinate') else fruit2.z)
        dx_time = self.traveling_time(dx, self.vx_max, self.ax_max)
        dz_time = self.traveling_time(dz, self.vz_max, self.az_max)
        return max(dx_time, dz_time)

    def draw_trajectories(self, video=False, video_name="trajectories.mp4", show_trajectory=True):
        """
        Draw the trajectories of the arms.
        :param video: whether to save the video
        :param video_name: name of the video file
        :param show_trajectory: whether to show the trajectory plot
        """
        print(f"Drawing trajectories...")
        self.show_trajectory = show_trajectory
        fig = plt.figure(figsize=(10, 8))
        self.ax = plt.axes()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-0.1, self.length+0.1)
        self.ax.set_ylim(-0.1, self.height+0.1)
        self.color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.draw_fruits()
        # Draw frame boundaries
        self.ax.plot([0, self.length, self.length, 0, 0], [0, 0, self.height, self.height, 0], color='black', linewidth=2)

        self.draw_init_arms()

        if video:
            max_timestamps = int(self.arms_plan.finishing_time / self.dt)
            ani = animation.FuncAnimation(fig, self.animation_update, frames=max_timestamps, interval=50, repeat=True)
            ani.save(video_name, writer='ffmpeg', fps=20)
        else:
            for r in range(self.arms_n):
                self.ax.plot(self.arms_plan.paths_x[r, :], self.arms_plan.paths_z[r, :], color=self.color_list[r % len(self.color_list)], label=f'Arm {r+1} Trajectory', linewidth=r+1)
                # Show some information
                harvest_number = len(self.arms_plan.harvesting_sequences[r])
                finish_time = self.arms_plan.arriving_times[r][-1] + self.harvesting_times[self.arms_plan.harvesting_sequences[r][-1]] if harvest_number > 0 else 0
                arm_FPT = harvest_number / finish_time if finish_time > 0 else 0
                finish_time_steps = round(finish_time / self.dt)
                total_travel_distance = np.sum([np.sqrt((self.arms_plan.paths_x[r, t+1]-self.arms_plan.paths_x[r, t])**2 + (self.arms_plan.paths_z[r, t+1]-self.arms_plan.paths_z[r, t])**2) for t in range(finish_time_steps-1)])
                print(f"Arm {r+1}: Harvested {harvest_number} fruits, Finishing time: {finish_time:.3f} s, FPT: {arm_FPT:.2f} fruits/s, Total travel distance: {total_travel_distance:.3f} m.")
    
    def draw_fruits(self):
        self.fruits_plot = [[self.ax.scatter(self.fruits[self.arms_plan.harvesting_sequences[r][k]].coordinate.x,
                                              self.fruits[self.arms_plan.harvesting_sequences[r][k]].coordinate.z, s=25, marker="o", color="black") for k in range(len(self.arms_plan.harvesting_sequences[r]))] for r in range(self.arms_n)]
    
    def draw_init_arms(self):
        self.gripper_plot = [self.ax.scatter(self.initial_positions[r].x, self.initial_positions[r].z, s=100, marker="1", color=self.color_list[r % len(self.color_list)]) for r in range(self.arms_n)]

    def animation_update(self, i):
        cur_time = i * self.dt
        self.update_arms_and_fruits(i)
        if self.show_trajectory:
            self.update_trajectories(i)
        cur_min_distance_between_arms = min([self.arms_plan.paths_z[r+1, i] - self.arms_plan.paths_z[r, i] for r in range(self.arms_n-1)]) if self.arms_n > 1 else float('inf')
        plt.title(f"Time: {cur_time:.2f} s. Finishing time: {self.arms_plan.finishing_time:.2f} s. Min distance between arms: {cur_min_distance_between_arms:.2f} m.")

    def update_arms_and_fruits(self, time_step):
        cur_time = time_step * self.dt
        for r in range(self.arms_n):
            # Draw the rest of the fruits
            for k in range(len(self.arms_plan.harvesting_sequences[r])):
                change_color_time = 0.0 if k == 0 else self.arms_plan.arriving_times[r][k-1]
                disappear_time = self.arms_plan.arriving_times[r][k] + self.harvesting_times[self.arms_plan.harvesting_sequences[r][k]]/2
                if change_color_time <= cur_time <= change_color_time + self.dt*2:
                    self.fruits_plot[r][k].set_color(self.color_list[r % len(self.color_list)])
                if disappear_time <= cur_time <= disappear_time + self.dt*2:
                    self.fruits_plot[r][k].set_alpha(0.0)
            # Update gripper position
            self.gripper_plot[r].set_offsets([self.arms_plan.paths_x[r,time_step], self.arms_plan.paths_z[r,time_step]])

    def update_trajectories(self, time_step):
        for r in range(self.arms_n):
            self.ax.plot(self.arms_plan.paths_x[r, max(time_step-1, 0):time_step+1],
                         self.arms_plan.paths_z[r, max(time_step-1, 0):time_step+1],
                         color=self.color_list[r % len(self.color_list)], linewidth=r+1)
            
    def zone_splitting_planning(self, gap=0.01):
        fruit_indices = [i for i in range(len(self.fruits))]
        fruit_sorted_by_z = sorted(fruit_indices, key=lambda fruit_idx: self.fruits[fruit_idx].coordinate.z)
        row_indices = self.split_list(fruit_sorted_by_z, self.arms_n, sort_horizontal=True)
        self.plan_on_row_indices(row_indices, gap=gap)

    def split_list(self, lst, num_parts, sort_horizontal=False):
        """
        Return the split list. It could be sorted horizontally considering the picking time plus the x coordinate of the fruits.
        """
        base_size = len(lst) // num_parts
        remainder = len(lst) % num_parts
        
        result = []
        idx = 0
        
        for i in range(num_parts):
            part_size = base_size + (1 if i < remainder else 0)
            cur_part = lst[idx:idx+part_size]
            if sort_horizontal:
                cur_part.sort(key=lambda fruit_idx: self.fruits[fruit_idx].coordinate.x)
            else:
                cur_part.sort(key=lambda fruit_idx: self.fruits[fruit_idx].coordinate.z)
            result.append(cur_part)
            idx += part_size
        
        return result
    
    def plan_on_row_indices(self, sequences, gap=0.01):
        finishing_time, arriving_times = self.heuristic_schedule_on_sequences(sequences)
        self.arms_plan = ArmsPlan(harvesting_sequences=sequences, arriving_times=arriving_times, finishing_time=finishing_time)
        assert self.trajectory_plan_on_sequences(self.arms_plan, alpha=0.9, use_makespan_as_constraint=False, log_to_console=True, gap=gap), "Final trajectory is infeasible!"

    def heuristic_schedule_on_sequences(self, sequences):
        """
        Heuristic scheduling on the given sequences.
        """
        R = range(self.arms_n)
        arriving_times = [[] for r in R]
        finishing_times = [0.0 for r in R]
        for r in R:
            cur_time = 0.0
            for k in range(len(sequences[r])):
                fruit_idx = sequences[r][k]
                travel_time = 0.0
                if k == 0:
                    travel_time = max(self.traveling_time(abs(self.initial_positions[r].x - self.fruits[fruit_idx].coordinate.x), self.vx_max, self.ax_max), self.traveling_time(abs(self.initial_positions[r].z - self.fruits[fruit_idx].coordinate.z), self.vz_max, self.az_max))
                else:
                    prev_fruit_idx = sequences[r][k-1]
                    travel_time = self.travel_time_f2f(self.fruits[prev_fruit_idx], self.fruits[fruit_idx])
                cur_time += travel_time
                # Ensure the fruit is detached
                arriving_times[r].append(cur_time)
                cur_time += self.harvesting_times[fruit_idx]
            finishing_times[r] = cur_time
        overall_finishing_time = max(finishing_times)
        return overall_finishing_time, arriving_times
    
    def plot_initial_fruits_and_arms(self):
        """
        Plot the initial fruits and arms positions.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.1, self.length+0.1)
        ax.set_ylim(-0.1, self.height+0.1)
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Draw fruits
        for fruit in self.fruits:
            ax.scatter(fruit.coordinate.x, fruit.coordinate.z, s=25, marker="o", color="black")
        # Draw frame boundaries
        ax.plot([0, self.length, self.length, 0, 0], [0, 0, self.height, self.height, 0], color='black', linewidth=2)
        # Draw initial arm positions
        for r in range(self.arms_n):
            ax.scatter(self.initial_positions[r].x, self.initial_positions[r].z, s=100, marker="1", color=color_list[r % len(color_list)])
        plt.title("Initial Fruits and Arms Positions")
        plt.show()



class Coordinate:    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FruitInfo:
    def __init__(self, x, y, z, detach_time):
        self.coordinate = Coordinate(x, y, z)
        self.detach_time = detach_time

class ArmsPlan:
    def __init__(self, moving_speed=0, harvesting_sequences=None, arriving_times=None, paths_x=None, paths_y=None, paths_z=None, finishing_time=float('inf'), harvesting_amount=0):
        self.moving_speed = moving_speed
        self.harvesting_sequences = harvesting_sequences
        self.arriving_times = arriving_times
        self.paths_x = paths_x
        self.paths_y = paths_y
        self.paths_z = paths_z
        self.finishing_time = finishing_time
        self.harvesting_amount = harvesting_amount

    def __lt__(self, other):
        return (self.finishing_time, -self.moving_speed) < (other.finishing_time, -other.moving_speed)

    def __repr__(self) -> str:
        return f"ArmsPlan(moving_speed={self.moving_speed}, \nfinishing_time={self.finishing_time}, harvest amount={self.harvesting_amount}\n\nHarvesting_sequences={self.harvesting_sequences}.\nArriving_times={self.arriving_times})."


if __name__ == "__main__":
    # Example usage
    hheight = 1.2
    length = 1.2
    arms_n = 3
    vx_max = 0.5
    vy_max = 0.5
    vz_max = 0.5
    ax_max = 1.0
    ay_max = 1.0
    az_max = 1.0
    dt = 0.1
    safety_distance = 0.1
    # Generate random fruits for testing
    rng = np.random.default_rng(42)
    fruit_n = 10
    depths = [0.2, 0.5]
    X = rng.uniform(0.0, length, fruit_n) # length
    Y = rng.uniform(depths[0], depths[1], fruit_n)  # depth
    Z = rng.uniform(safety_distance, height - safety_distance, fruit_n)  # height
    detach_times = rng.uniform(2.0, 3.0, fruit_n)  # detach times, it could be a function of ripening levels or it could be a constant.
    fruits = [FruitInfo(X[i], Y[i], Z[i], detach_times[i]) for i in range(fruit_n)]
    model = StaticBilevelModel(fruits, height, length, arms_n, vx_max, vy_max, vz_max, ax_max, ay_max, az_max, dt, safety_distance)
    model.plot_initial_fruits_and_arms()
    tic = time.time()
    # model.bilevel_trajectory_planning() # Not working for now
    model.zone_splitting_planning(gap=0.05) # Larger gap for faster solving
    toc = time.time()
    print(f"Time taken for optimization: {toc - tic:.3f} seconds")
    print("Optimization completed.")
    model.draw_trajectories(video=False, show_trajectory=True)
    print(f"Model results are:\n{model.arms_plan}")
    print(f"Trajectories are stored in model.arms_plan.paths_x, paths_y, paths_z.")
    