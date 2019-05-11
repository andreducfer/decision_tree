from src.data_handler import CommandLine, Instance
from src.decision_tree import Solution, Greedy

import time
import sys

if __name__ == "__main__":
    # datasets = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10']
    datasets = ['p10']
    for dataset in datasets:
        cmd = CommandLine(sys.argv + [dataset])
    
        # Initialization of the problem data from the commandline
        instance = Instance(cmd.instance_path, cmd.solution_path, cmd.seed, cmd.max_depth, cmd.cpu_time)

        # Initialization of a solution structure
        solution = Solution(instance)

        # Run the greedy algorithm
        print("----- STARTING DECISION TREE OPTIMIZATION")
        instance.start_time = time.time()
        solver = Greedy(instance, solution)
        # solver.run()
        solver.local_search()
        instance.end_time = time.time()

        delta = instance.end_time - instance.start_time
        print("----- DECISION TREE OPTIMIZATION COMPLETED IN " + str(delta) + " (s)")

        # Printing the solution and exporting statistics (also export results into a file)
        solution.print_and_export(cmd.solution_path)
        print("----- END OF ALGORITHM")