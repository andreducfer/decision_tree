from data_handler import Instance
from decision_tree import Solution, Greedy

import time
import argparse
from os import listdir
from os.path import join
from datetime import datetime
import random

default_seed_list = [12, 0, 25, 48, 998]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decision tree optimization arguments.')

    parser.add_argument('-i', '--datasets_dir', default='dataset', help="Directory containing dataset files.")
    parser.add_argument('-o', '--solution_dir', default='solution', help="Directory to save solutions.")
    parser.add_argument('-s', '--seed', type=int, nargs='+', default=default_seed_list, help='Seed for the random function.')
    parser.add_argument('-d', '--max_depth', default=4, type=int, help="Maximum level of tree.")
    parser.add_argument('-m', '--max_execution_time_in_seconds', default=300, type=int, help="Maximum execution time for each instance. In seconds.")
    parser.add_argument('-f', '--dataset_file', help="Dataset file to be processed.")

    args = parser.parse_args()

    if args.dataset_file is not None:
        dataset_files = [args.dataset_file]
    else:
        dataset_files = sorted(listdir(args.datasets_dir))

    for dataset_file in dataset_files:
        for seed in args.seed:
            random.seed(seed)

            # Path where is the dataset
            dataset_path = join(args.datasets_dir, dataset_file)

            print("Selected dataset '%s'" % dataset_file)
            print("Parameters:")
            print("seed: %d\nmax_depth: %d\nmax_execution_time_in_seconds %d" % (seed, args.max_depth, args.max_execution_time_in_seconds))

            time_limit = time.time() + args.max_execution_time_in_seconds

            # Path and model of solution file
            time_now = datetime.now().strftime("%Y%m%d%H%M%S")
            solution_file = "%s-%s-seed_%d.txt" % (dataset_file.rstrip('.txt'), time_now, seed)
            solution_path = join(args.solution_dir, solution_file)

            # Initialization of instance
            instance = Instance(dataset_path, solution_path, seed, args.max_depth, args.max_execution_time_in_seconds)

            # Initialization of a solution structure
            solution = Solution(instance)

            print("\n----- STARTING DECISION TREE OPTIMIZATION...")
            instance.start_time = time.time()
            solver = Greedy(instance, solution)

            # Run the greedy or iterated local search algorithm
            # solver.run()
            solver.iterated_local_search(time_limit)

            # Write csv files with the results of accuracy and misclassifieds
            solution.write_statistics_file()

            instance.end_time = time.time()

            delta = instance.end_time - instance.start_time
            print("\n----- DECISION TREE OPTIMIZATION COMPLETED IN " + str(delta) + " (s)")

            # Printing the solution and exporting statistics (also export results into a file)
            solution.print_and_export(instance.solution_path)
            print("\n----- END OF ALGORITHM")

    # Solution.statistical_test()