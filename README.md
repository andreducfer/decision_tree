# Decision Tree Optimization (Iterated Local Search)

Decision tree algorithm optimized with iterated local search. 

### Terminal call

To call the algorithm using terminal execute the following command:

    python3 main.py 

Optional arguments:

* -h => Show help message and exit (Optional)

* -i => Directory containing dataset files. default='dataset'
                        
* -o => Directory to save solutions. default='solution'
                        
* -s => List of seeds for the random function. default=[12, 0, 25, 48, 998]
                        
* -d => Maximum level of tree. default=4
                        
* -m => Maximum execution time for each instance in seconds. default=300
                        
* -f => Dataset file to be processed. If not filled, execute for all instances.

Example:

    python3 main.py -f p08.txt -s 23 18 -m 30
    
With this command the algorithm will run for 30 seconds for every seed (23, 18)
for the dataset p08.txt


### Statistical test
To a complete statistical result at the end, follow the next steps:

* On _main.py_, uncomment the call _solver.run()_
* On _main.py_, comment the call _solver.iterated_local_search(time_limit)_
* On _main.py_, comment the call _Solution.statistical_test()_
* Delete all files inside the _solution_ folder
* Run the code

After that:

* On _main.py_, comment the call _solver.run()_
* On _main.py_, uncomment the call _solver.iterated_local_search(time_limit)_
* On _main.py_, uncomment the call _Solution.statistical_test()_
* Run the code

At the end, the algorithm will generate a file with the name _wilcoxon_result.txt_
with the result of the statistical test for the algorithm.               