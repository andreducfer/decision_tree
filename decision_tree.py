from data_handler import Instance

import enum
import numpy as np
import time
import random
from scipy import stats
import csv

class NodeType(enum.Enum):
    NULL = 0
    INTERNAL = 1
    LEAF = 2


class Node:

    def __init__(self, instance):
        if not isinstance(instance,  Instance):
            raise TypeError("ERROR: instance variable is not data_handler.Instance. Type: " + type(instance))

        self.instance = instance                                     # Access to the problem and dataset parameters
        self.node_type = NodeType.NULL                               # Node type 
        self.split_attribute_id = -1                                 # Attribute to which the split is applied
        self.split_attribute_value = float("-inf")                   # Threshold value for the split
        self.samples = []                                            # Samples from the training set at this node
        self.num_samples_per_class = np.zeros(instance.num_classes)  # Number of samples of each class at this node
        self.num_samples = 0                                         # Total number of samples in this node
        self.majority_class_id = -1                                  # Majority class in this node
        self.num_majority_class = 0                                  # Number of elements in the majority class
        self.entropy = float("-inf")                                 # Entropy in this node
        self.level = 0                                               # Level of the node on the tree

        #TO SAVE A LIST OF BEST ATTRIBUTES TO SPLIT NOT USED FOR THE CART ALGORITHM(NEED THE ID AND VALUE OF ATTRIBUTE)
        self.possible_splits = np.zeros([30,2])
        self.possible_gains = np.zeros(30)
        self.exist_split = np.zeros(30, dtype=bool)

        # TO MEASURE CPU TIME
        self.start_time = 0                                          # Time when the algorithm started
        self.end_time = 0                                            # Time when the algorithm ended


    def evaluate(self):
        frac = self.num_samples_per_class / (self.num_samples + 0.000001) # Added this small value in a case that possible can have 0 num_samples
        frac = frac[self.num_samples_per_class > 0]
        parcial = frac * np.log2(frac)
        self.entropy = -parcial.sum()
        self.majority_class_id = self.num_samples_per_class.argmax()
        self.num_majority_class = self.num_samples_per_class[self.majority_class_id]


    def add_sample(self, sample_id):
        self.samples.append(sample_id)
        self.num_samples_per_class[self.instance.data_classes[sample_id]] += 1
        self.num_samples += 1        


class Solution:

    def __init__(self, instance):
        if not isinstance(instance,  Instance):
            raise TypeError("ERROR: instance variable is not data_handler.Instance. Type: " + type(instance))
        self.instance = instance # Access to the problem and dataset parameters
        self.tree = [Node(instance) for _ in range(2**(instance.max_depth+1)-1)]
        self.tree[0].node_type = NodeType.LEAF
        for i in range(instance.num_samples):
            self.tree[0].add_sample(i)
        self.tree[0].evaluate()


    # Function that return the number os misclassifieds of the tree
    def get_num_misclassified_samples(self):
        num_misclassified_samples = 0

        for d in range(self.instance.max_depth + 1):
            for i in range(2 ** d - 1, 2 ** (d + 1) - 1):
                if self.tree[i].node_type == NodeType.LEAF:
                    num_misclassified = self.tree[i].num_samples - self.tree[i].num_majority_class
                    num_misclassified_samples += num_misclassified

        return num_misclassified_samples


    # Function that returns the accuracy of the solution
    def get_accuracy(self):
        num_misclassified = self.get_num_misclassified_samples()
        num_samples = self.instance.num_samples

        accuracy = (num_samples - num_misclassified) / num_samples * 100

        return accuracy


    def print_and_export(self, filename=None):
        num_misclassified_samples = 0
        accuracy = self.get_accuracy()
        tree = []

        # Print solution
        print("\n---------------------------------------- PRINTING SOLUTION ----------------------------------------")        
        for d in range(self.instance.max_depth + 1):
            # Printing one complete level of the tree
            level_info = ""
            for i in range(2**d - 1, 2**(d+1) - 1):
                if self.tree[i].node_type == NodeType.INTERNAL:
                    attribute_id = self.tree[i].split_attribute_id
                    attribute_value = self.tree[i].split_attribute_value
                    attribute_type = self.instance.attribute_types[attribute_id]

                    node_template = "(N%d,A[%d]%s" + ("%f" if attribute_type == 'N' else "%d") + ") "
                    node_info = node_template % (i, attribute_id,
                                                 '<=' if attribute_type == 'N' else '=',
                                                 attribute_value)
                elif self.tree[i].node_type == NodeType.LEAF:
                    num_misclassified = self.tree[i].num_samples - self.tree[i].num_majority_class
                    num_misclassified_samples += num_misclassified

                    node_template = "(L%d,C%d,%d,%d) "
                    node_info = node_template % (i, self.tree[i].majority_class_id,
                                                 self.tree[i].num_majority_class,
                                                 num_misclassified)
                else:
                    continue
                level_info += node_info
            print(level_info)
            tree.append(level_info)
        print("%d/%d MISCLASSIFIED SAMPLES" % (num_misclassified_samples, self.instance.num_samples))
        print("\n%d ACCURACY\n" % accuracy)
        print("---------------------------------------------------------------------------------------------------\n")

        if filename is not None:
            # Dump result
            with open(filename, mode='w') as fp:
                delta = self.instance.end_time - self.instance.start_time
                fp.write("NUMBER OF SAMPLES: " + str(self.instance.num_samples) + "\n")
                fp.write("NUMBER OF ATTRIBUTES: " + str(self.instance.num_attributes) + "\n")
                fp.write("NUMBER OF CLASSES: " + str(self.instance.num_classes) + "\n")
                fp.write("TIME(s): " + str(delta) + "\n")
                fp.write("NB_SAMPLES: " + str(self.instance.num_samples) + "\n")
                fp.write("NB_MISCLASSIFIED: " + str(num_misclassified_samples) + "\n")
                fp.write("ACCURACY: " + str(accuracy) + "%\n")
                fp.write("TREE: \n")
                for level_tree in tree:
                    fp.write(level_tree + "\n")


    def write_statistics_file(self):
        accuracy_file = "solution/statistics_accuracy.csv"
        misclassified_file = "solution/statistics_misclassified.csv"
        accuracy = []
        misclassified = []
        accuracy.append(round(float(self.get_accuracy()), 2))
        misclassified.append(round(float(self.get_num_misclassified_samples()), 2))

        with open (accuracy_file, 'a') as new_file:
            csv_writer = csv.writer(new_file, delimiter=',')
            csv_writer.writerow(accuracy)

        with open (misclassified_file, 'a') as new_file:
            csv_writer = csv.writer(new_file, delimiter=',')
            csv_writer.writerow(misclassified)


    @classmethod
    def statistical_test(self):
        accuracy_file = "solution/statistics_accuracy.csv"
        wilcoxon_file = 'solution/wilcoxon_result.txt'
        accuracy_list = []
        greedy_list = []
        iterated_list = []

        with open(accuracy_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for line in csv_reader:
                accuracy_list.append(line[0])

        accuracy_list = list(map(float, accuracy_list))

        for index, value in enumerate(accuracy_list):
            if index < len(accuracy_list)/2:
                greedy_list.append(value)
            else:
                iterated_list.append(value)

        wilcoxon_result = stats.wilcoxon(greedy_list, iterated_list, zero_method="wilcox", correction=True)
        print(str(wilcoxon_result))

        with open(wilcoxon_file, mode='w') as fp:
            fp.write(str(wilcoxon_result))


class Greedy:

    def __init__(self, instance, solution):
        if not isinstance(instance,  Instance):
            raise TypeError("ERROR: instance variable is not data_handler.Instance. Type: " + type(instance))
        if not isinstance(solution,  Solution):
            raise TypeError("ERROR: solution variable is not decision_tree.Solution. Type: " + type(solution))
        self.instance = instance
        self.solution = solution
        self.possible_nodes_to_perturbe = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) # Possible nodes to use in perturbation


    def run(self):
        self._recursive_construction(0, 0)


    # CART solution
    def _recursive_construction(self, node_id, level):
        # BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS
        num_majority_class = self.solution.tree[node_id].num_majority_class
        num_samples = self.solution.tree[node_id].num_samples
        if level >= self.instance.max_depth or num_majority_class == num_samples:
            self.solution.tree[node_id].level = level
            return

        # LOOK FOR A BEST SPLIT
        all_identical = True # To detect contradictory data
        parent_entropy = self.solution.tree[node_id].entropy
        best_gain = float("-inf")
        best_split_attribute_id = -1
        best_split_threhold = float("-inf")

        for attr_id, attr_type in enumerate(self.instance.attribute_types):
            # Define some data structures                
            sample_ids = self.solution.tree[node_id].samples
            values = self.instance.data_attributes[sample_ids, attr_id]
            class_ids = self.instance.data_classes[sample_ids]

            if attr_type == 'N':
                # CASE 1) -- FIND SPLIT WITH BEST INFORMATION GAIN FOR NUMERICAL ATTRIBUTE c                

                # Store the possible levels of this attribute among the samples (will allow to "skip" samples with equal attribute value)
                unique_values = np.unique(values)
                
                # If all sample have the same level for this attribute, it's useless to look for a split
                if unique_values.size <= 1:
                    continue
                all_identical = False

                # Order of the samples according to attribute c                
                sorted_positions = np.argsort(values)
                sorted_values = values[sorted_positions]
                sorted_class_ids = class_ids[sorted_positions]
                
                # Initially all samples are on the right
                num_samples_per_class_left = np.zeros(self.instance.num_classes, dtype=np.int)
                num_samples_per_class_right = np.copy(self.solution.tree[node_id].num_samples_per_class)                
                
                # Go through all possible attribute values in increasing order
                sample_idx = 0
                for threshold in unique_values:
                    # Iterate on all samples with this unique_values and switch them to the left
                    while sample_idx < num_samples and  sorted_values[sample_idx] < threshold + 0.000001:
                        class_id = sorted_class_ids[sample_idx]
                        num_samples_per_class_left[class_id] += 1
                        num_samples_per_class_right[class_id] -= 1
                        sample_idx += 1
                    
                    # No need to consider the case in which all samples have been switched to the left
                    if sample_idx == num_samples:
                        continue
                    
                    frac_left = num_samples_per_class_left / sample_idx
                    frac_left = frac_left[num_samples_per_class_left > 0]
                    parcial_left = frac_left * np.log2(frac_left)
                    entropy_left = -parcial_left.sum()

                    frac_right = num_samples_per_class_right / (num_samples - sample_idx)
                    frac_right = frac_right[num_samples_per_class_right > 0]
                    parcial_right = frac_right * np.log2(frac_right)
                    entropy_right = -parcial_right.sum()

                    # Evaluate the information gain and store if this is the best option found until now
                    gain = parent_entropy - (sample_idx * entropy_left + (num_samples - sample_idx) * entropy_right) / num_samples
                    if gain > best_gain:
                        best_gain = gain
                        best_split_threhold = threshold
                        best_split_attribute_id = attr_id
                        self.solution.tree[node_id].level = level
                    else:
                        # To fill a list with other possible splits to use on the local search and perturbation
                        index_min_gain = np.argmin(self.solution.tree[node_id].possible_gains)
                        if(gain > self.solution.tree[node_id].possible_gains[index_min_gain]):
                            self.solution.tree[node_id].possible_gains[index_min_gain] = gain
                            self.solution.tree[node_id].exist_split[index_min_gain] = True
                            self.solution.tree[node_id].possible_splits[index_min_gain, 0] = attr_id
                            self.solution.tree[node_id].possible_splits[index_min_gain, 1] = threshold
            else:
                # CASE 2) -- FIND BEST SPLIT FOR CATEGORICAL ATTRIBUTE c
                num_level = self.instance.num_levels[attr_id]
                num_classes = self.instance.num_classes
                levels = values.astype(np.int)

                num_samples_per_level = np.bincount(levels, minlength=num_level)
                num_samples_per_class = np.bincount(class_ids, minlength=num_classes)
                num_samples_per_class_level = np.zeros((num_classes, num_level), dtype=np.int)
                np.add.at(num_samples_per_class_level, [class_ids, levels], 1)
                
                for l in range(num_level):
                    if num_samples_per_level[l] < 1 or num_samples_per_level[l] >= num_samples:
                        continue
                    all_identical = False
                    
                    frac_level = num_samples_per_class_level[:, l] / num_samples_per_level[l]
                    frac_level = frac_level[num_samples_per_class_level[:, l] > 0]
                    parcial_level = frac_level * np.log2(frac_level)
                    entropy_level = -parcial_level.sum()

                    frac_others = (num_samples_per_class - num_samples_per_class_level[:, l]) / (num_samples - num_samples_per_level[l])
                    frac_others = frac_others[(num_samples_per_class - num_samples_per_class_level[:, l]) > 0]
                    parcial_others = frac_others * np.log2(frac_others)
                    entropy_others = -parcial_others.sum()

                    gain = parent_entropy - (num_samples_per_level[l] * entropy_level + (num_samples - num_samples_per_level[l]) * entropy_others) / num_samples
                    if gain > best_gain:
                        best_gain = gain
                        best_split_threhold = l
                        best_split_attribute_id = attr_id
                        self.solution.tree[node_id].level = level
                    else:
                        # To fill a list with other possible splits to use on the local search and perturbation
                        index_min_gain = np.argmin(self.solution.tree[node_id].possible_gains)
                        if(gain > self.solution.tree[node_id].possible_gains[index_min_gain]):
                            self.solution.tree[node_id].possible_gains[index_min_gain] = gain
                            self.solution.tree[node_id].exist_split[index_min_gain] = True
                            self.solution.tree[node_id].possible_splits[index_min_gain, 0] = attr_id
                            self.solution.tree[node_id].possible_splits[index_min_gain, 1] = l

        # SPECIAL CASE TO HANDLE POSSIBLE CONTRADICTIONS IN THE DATA
        # (Situations where the same samples have different classes -- In this case no improving split can be found)
        if all_identical: 
            return

        # Make the split on the tree
        self._split(node_id, level, best_split_attribute_id, best_split_threhold)


    # Function that make the split on the tree
    def _split(self, node_id, level, best_split_attribute_id, best_split_threhold):
        # APPLY THE SPLIT AND RECURSIVE CALL
        self.solution.tree[node_id].split_attribute_id = best_split_attribute_id
        self.solution.tree[node_id].split_attribute_value = best_split_threhold
        self.solution.tree[node_id].node_type = NodeType.INTERNAL
        self.solution.tree[2*node_id+1].node_type = NodeType.LEAF
        self.solution.tree[2*node_id+2].node_type = NodeType.LEAF
        for sample_id in self.solution.tree[node_id].samples:
            attribute_type = self.instance.attribute_types[best_split_attribute_id]
            attribute_value = self.instance.data_attributes[sample_id, best_split_attribute_id]
            if (attribute_type == 'N' and attribute_value <= best_split_threhold) or (attribute_type == 'C' and attribute_value == best_split_threhold):
                self.solution.tree[2*node_id+1].add_sample(sample_id)
            else:
                self.solution.tree[2*node_id+2].add_sample(sample_id)
        self.solution.tree[2*node_id+1].evaluate() # Setting all other data structures
        self.solution.tree[2*node_id+2].evaluate() # Setting all other data structures
        self._recursive_construction(2*node_id+1,level+1) # Recursive call
        self._recursive_construction(2*node_id+2,level+1) # Recursive call


    # Function used to clear the children of the current node
    def _zero_nodes(self, node_id, level):
        if node_id < len(self.solution.tree):
            self.solution.tree[node_id] = Node(self.instance)

            self._zero_nodes(2 * node_id + 1, level + 1)
            self._zero_nodes(2 * node_id + 2, level + 1)


    # Return a list of nodes that contain samples
    def _nodes_with_sample(self):
        list_possible_nodes = []
        for node in self.possible_nodes_to_perturbe:
            num_samples = self.solution.tree[node].num_samples
            num_majority_class = self.solution.tree[node].num_majority_class

            if num_samples > 1 and num_samples > num_majority_class:
                list_possible_nodes.append(node)

        return list_possible_nodes


    # Change the split in 3 different nodes to make a perturbation.
    def _perturbation(self):

        # Initialize 3 nodes with 0 in case that don't have 3 different nodes to split.
        random_node_one = 0
        random_node_two = 0
        random_node_three = 0

        # Take only the nodes with sample to make the splits and randomly choose 3
        nodes_with_samples = self._nodes_with_sample()
        nodes_to_split = random.choices(nodes_with_samples, k = 3)
        nodes_to_split.sort()

        # To garantee that will do the splits from lowest to highest
        random_node_one = nodes_to_split[0]
        random_node_two = nodes_to_split[1]
        random_node_three = nodes_to_split[2]

        # Take the level of the selected nodes on the tree
        level_node_one = self.solution.tree[random_node_one].level
        level_node_two = self.solution.tree[random_node_two].level
        level_node_three = self.solution.tree[random_node_three].level

        # Choose randomly the splits for every one of the selected nodes, based on the list of possible splits
        random_split_one = random.choice(self.solution.tree[random_node_one].possible_splits)
        random_split_two = random.choice(self.solution.tree[random_node_two].possible_splits)
        random_split_three = random.choice(self.solution.tree[random_node_three].possible_splits)

        # Clear the children of the  node one and make a split
        self._zero_nodes(2 * random_node_one + 1, level_node_one + 1)
        self._zero_nodes(2 * random_node_one + 2, level_node_one + 1)
        self._split(random_node_one, level_node_one, int(random_split_one[0]), random_split_one[1])

        # Clear the children of the node two and make a split
        self._zero_nodes(2 * random_node_two + 1, level_node_two + 1)
        self._zero_nodes(2 * random_node_two + 2, level_node_two + 1)
        self._split(random_node_two, level_node_two, int(random_split_two[0]), random_split_two[1])

        # Clear the children of the node three and make a split
        self._zero_nodes(2 * random_node_three + 1, level_node_three + 1)
        self._zero_nodes(2 * random_node_three + 2, level_node_three + 1)
        self._split(random_node_three, level_node_three, int(random_split_three[0]), random_split_three[1])

        # Return the minor node to start local search from this node and don't need to recalculate all the tree.
        return random_node_one


    # Do a local search iterating throw the list of possible splits
    def _local_search(self, minor_node):

        # Initial tree used to make a best improvement (always returning the initial tree to the solution tree when change the node and when change the split)
        initial_tree = np.copy(self.solution.tree)
        # Tree that will be updated if find a better solution
        local_tree = np.copy(initial_tree)
        local_misclassified = self.solution.get_num_misclassified_samples()

        if local_misclassified > 0:
            num_nodes = len(self.solution.tree)
            # Iterate throw the nodes of the tree, starting from the minor node changed on the perturbation, so don't need to recalculate all the tree
            for node_id in range(minor_node, num_nodes):
                # Restore the initial tree for every new node for a best improvement
                self.solution.tree = np.copy(initial_tree)
                level = self.solution.tree[node_id].level

                if self.solution.tree[node_id].node_type == NodeType.INTERNAL:
                    for i in range(len(self.solution.tree[node_id].possible_gains)):
                        # Verify if there is a possible split in the list(in case that don't fill all the 30 positions)
                        if self.solution.tree[node_id].exist_split[i]:
                            # Restore the initial tree for every new possible split for a best improvement
                            self.solution.tree = np.copy(initial_tree)

                            # Clear the children of the current node
                            self._zero_nodes(2 * node_id + 1, level + 1)
                            self._zero_nodes(2 * node_id + 2, level + 1)

                            # Split that will be used to try a better tree
                            split_attribute_id = int(self.solution.tree[node_id].possible_splits[i, 0])
                            split_threhold = self.solution.tree[node_id].possible_splits[i, 1]

                            # Do the split and verify the number of misclassifieds after this split
                            self._split(node_id, level, split_attribute_id, split_threhold)
                            split_misclassified = self.solution.get_num_misclassified_samples()

                            # Here accept to update de local tree if it has less or the same number of misclassifieds to walk more on the possible solutions,
                            # and not only if it has less misclassifieds.
                            if split_misclassified <= local_misclassified:
                                local_tree = np.copy(self.solution.tree)
                                local_misclassified = split_misclassified

        # At the end update the solution tree with the local tree, that is the best tree after the local search or at least the same that started
        self.solution.tree = local_tree


    def iterated_local_search(self, time_limit):

        # Initial solution
        self._recursive_construction(0, 0)
        greedy_misclassifieds = self.solution.get_num_misclassified_samples()

        print("\n+++++++++++++++++++++++++++++++INITIAL ITERATED_SEARCH+++++++++++++++++++++++++++++++\n")

        # If the CART solution returned a tree with 0 misclassifieds it is not necessary to improve the tree
        if greedy_misclassifieds > 0:
            print("Performing local search...")

            # Do a local search on the initial solution
            self._local_search(0)

            # Tree that will be updated if find a better solution
            best_tree = np.copy(self.solution.tree)
            best_misclassifieds = self.solution.get_num_misclassified_samples()

            print("Local search finished. Misclassifieds: %s " % best_misclassifieds)

            while(True):
                print("Performing perturbation...")

                # Do a perturbation on the best tree that was encountered on the local search
                minor_node = self._perturbation()

                print("Performing local search...")

                # Do a local search on the perturbed tree
                self._local_search(minor_node)
                local_misclassifieds = self.solution.get_num_misclassified_samples()

                print("Local search finished. Misclassifieds: %s " % local_misclassifieds)

                # Here accept to update de best tree if it has less or the same number of misclassifieds to walk more on the possible solutions,
                # and not only if it has less misclassifieds.
                if local_misclassifieds <= best_misclassifieds:
                    best_tree = np.copy(self.solution.tree)
                    best_misclassifieds = local_misclassifieds

                # Restore the best tree to the solution for a best improvement
                self.solution.tree = np.copy(best_tree)

                print("--------------------------")

                # To ensure that the algorithm will run for up to 5 minutes
                if (time.time() > time_limit) or (best_misclassifieds == 0):
                    break

            # Return the best tree after the algorithm run
            self.solution.tree = best_tree

        print("\n+++++++++++++++++++++++++++++++END ITERATED_SEARCH+++++++++++++++++++++++++++++++\n")




        # to_consider_greedy = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 71.04, 71.04, 71.04, 71.04, 71.04, 98.25, 98.25,
        #                       98.25, 98.25, 98.25, 79.81, 79.81, 79.81, 79.81, 79.81, 98.42, 98.42, 98.42, 98.42, 98.42, 85.86, 85.86, 85.86, 85.86,
        #                       85.86, 97.0, 97.0, 97.0, 97.0, 97.0, 83.68, 83.68, 83.68, 83.68, 83.68, 94.09, 94.09, 94.09, 94.09, 94.09]
        #
        # to_consider_iterated_local_search = [ 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 77.76, 77.76, 77.76, 77.60,
        #                                       77.92, 98.98, 98.83, 98.83, 98.98, 98.98, 81.55, 81.55, 81.68, 81.68, 81.42, 99.12, 99.12, 98.95,
        #                                       98.95, 99.47, 91.92, 92.42, 92.42, 92.42, 92.42, 98.57, 98.57, 98.57, 98.43, 98.57, 86.69, 86.69,
        #                                       86.69, 86.69, 86.69, 94.12, 95.12, 95.12, 95.12, 95.12]

        # wilcoxon_result = stats.wilcoxon(to_consider_greedy, to_consider_iterated_local_search, zero_method="wilcox", correction=True)
        #
        # print(str(wilcoxon_result))
