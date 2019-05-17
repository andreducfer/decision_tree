from data_handler import Instance

import enum
import numpy as np
import time
import random

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

        #TO SAVE LIST OF BEST ATTRIBUTES NOT USED TO SPLIT (WE NEED THE ID AND VALUE OU ATTRIBUTE)
        self.possible_splits = np.zeros([30,2])
        self.possible_gains = np.zeros(30)

        # TO MEASURE CPU TIME
        self.start_time = 0                                          # Time when the algorithm started
        self.end_time = 0                                            # Time when the algorithm ended

    def evaluate(self):
        frac = self.num_samples_per_class / (self.num_samples + 0.000001)
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


    def get_num_misclassified_samples(self):
        num_misclassified_samples = 0

        for d in range(self.instance.max_depth + 1):
            for i in range(2 ** d - 1, 2 ** (d + 1) - 1):
                if self.tree[i].node_type == NodeType.LEAF:
                    num_misclassified = self.tree[i].num_samples - self.tree[i].num_majority_class
                    num_misclassified_samples += num_misclassified

        return num_misclassified_samples


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


class Greedy:

    def __init__(self, instance, solution):
        if not isinstance(instance,  Instance):
            raise TypeError("ERROR: instance variable is not data_handler.Instance. Type: " + type(instance))
        if not isinstance(solution,  Solution):
            raise TypeError("ERROR: solution variable is not decision_tree.Solution. Type: " + type(solution))
        self.instance = instance
        self.solution = solution
        self.acceptable_deterioration_factor = 1.25
        self.index_nodes_left = np.array([1, 3, 4])
        self.index_nodes_right = np.array([2, 5, 6])
        self.index_nodes = np.array([0, 1, 2, 3, 4, 5, 6])

    def run(self):
        self._recursive_construction(0, 0)
    
    def _recursive_construction(self, node_id, level):
        # BASE CASES -- MAXIMUM LEVEL HAS BEEN ATTAINED OR ALL SAMPLES BELONG TO THE SAME CLASS
        num_majority_class = self.solution.tree[node_id].num_majority_class
        num_samples = self.solution.tree[node_id].num_samples
        if level >= self.instance.max_depth or num_majority_class == num_samples :
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
                        #Fill array with best splits not used
                        index_min_gain = np.argmin(self.solution.tree[node_id].possible_gains)
                        if(gain > self.solution.tree[node_id].possible_gains[index_min_gain]):
                            self.solution.tree[node_id].possible_gains[index_min_gain] = gain
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
                        index_min_gain = np.argmin(self.solution.tree[node_id].possible_gains)
                        if(gain > self.solution.tree[node_id].possible_gains[index_min_gain]):
                            self.solution.tree[node_id].possible_gains[index_min_gain] = gain
                            self.solution.tree[node_id].possible_splits[index_min_gain, 0] = attr_id
                            self.solution.tree[node_id].possible_splits[index_min_gain, 1] = l

        # SPECIAL CASE TO HANDLE POSSIBLE CONTADICTIONS IN THE DATA
        # (Situations where the same samples have different classes -- In this case no improving split can be found)
        if all_identical: 
            return

        self._split(node_id, level, best_split_attribute_id, best_split_threhold)


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


    def _zero_nodes(self, node_id, level):
        if node_id < len(self.solution.tree):
            self.solution.tree[node_id] = Node(self.instance)

            self._zero_nodes(2 * node_id + 1, level + 1)
            self._zero_nodes(2 * node_id + 2, level + 1)


    # Return only nodes on the left and right side of the tree that have samples, and is interisting to make perturbation
    # def _nodes_with_sample(self):
    #     left_nodes = []
    #     right_nodes = []
    #     for node_left in self.index_nodes_left:
    #         num_samples = self.solution.tree[node_left].num_samples
    #         num_majority_class = self.solution.tree[node_left].num_majority_class
    #
    #         if num_samples > 1 and num_samples > num_majority_class:
    #             left_nodes.append(node_left)
    #
    #     for node_right in self.index_nodes_right:
    #         num_samples = self.solution.tree[node_right].num_samples
    #         num_majority_class = self.solution.tree[node_right].num_majority_class
    #
    #         if num_samples > 1 and num_samples > num_majority_class:
    #             right_nodes.append(node_right)
    #
    #     return left_nodes, right_nodes


    def _nodes_with_sample(self):
        list_possible_nodes = []
        for node in self.index_nodes:
            num_samples = self.solution.tree[node].num_samples
            num_majority_class = self.solution.tree[node].num_majority_class

            if num_samples > 1 and num_samples > num_majority_class:
                list_possible_nodes.append(node)

        return list_possible_nodes


    # Change splits in 2 nodes at the same time
    def _perturbation(self):
        # random_node_left = 0
        # random_node_right = 0

        # # To see if we have samples in a node to make a perturbation using only nodes with samples
        # left_nodes, right_nodes = self._nodes_with_sample()
        #
        # if len(left_nodes) > 0 and len(right_nodes) > 0:
        #     random_node_left = random.choice(left_nodes)
        #     random_node_right = random.choice(right_nodes)
        # elif len(left_nodes) == 1 and len(right_nodes) == 0:
        #     random_node_right = left_nodes[0]
        # elif len(left_nodes) == 0 and len(right_nodes) == 1:
        #     random_node_right = right_nodes[0]
        # else:
        #     if len(right_nodes) >= 2:
        #         chosen = random.choices(right_nodes, k=2)
        #         random_node_left = min(chosen)
        #         random_node_right = max(chosen)
        #     elif(len(left_nodes) >=2):
        #         chosen = random.choices(left_nodes, k=2)
        #         random_node_left = min(chosen)
        #         random_node_right = max(chosen)

        nodes_with_samples = self._nodes_with_sample()
        nodes_to_split = random.choices(nodes_with_samples, k = 2)

        random_node_left = min(nodes_to_split)
        random_node_right = max(nodes_to_split)


        level_left = self.solution.tree[random_node_left].level
        level_right = self.solution.tree[random_node_right].level

        random_split_left = random.choice(self.solution.tree[random_node_left].possible_splits)
        random_split_right = random.choice(self.solution.tree[random_node_right].possible_splits)

        self._zero_nodes(2 * random_node_left + 1, self.solution.tree[random_node_left].level + 1)
        self._zero_nodes(2 * random_node_left + 2, self.solution.tree[random_node_left].level + 1)
        self._zero_nodes(2 * random_node_right + 1, self.solution.tree[random_node_right].level + 1)
        self._zero_nodes(2 * random_node_right + 2, self.solution.tree[random_node_right].level + 1)

        self._split(random_node_left, level_left, int(random_split_left[0]), random_split_left[1])
        self._split(random_node_right, level_right, int(random_split_right[0]), random_split_right[1])

        # Return only the minor node to start local search in this node
        return random_node_left if random_node_left <= random_node_right else random_node_right


    def _local_search(self, minor_node, annealing):
        initial_tree = np.copy(self.solution.tree)
        local_tree = np.copy(initial_tree)
        local_misclassified = self.solution.get_num_misclassified_samples()
        # initial_misclassified = np.copy(local_misclassified)

        if local_misclassified > 0:
            num_nodes = len(self.solution.tree)
            for node_id in range(minor_node, num_nodes):
                self.solution.tree = np.copy(initial_tree)
                level = self.solution.tree[node_id].level

                if self.solution.tree[node_id].node_type == NodeType.INTERNAL:
                    # print("###############################################")
                    # print("node_id: %d" % node_id)
                    # print("Level: " + str(level))
                    # print("Initial number of misclassifieds: " + str(local_misclassified))

                    for i in range(len(self.solution.tree[node_id].possible_gains)):
                        # print("===============================================")
                        # print("Attribute for current split: %d" % i)

                        # If this happens we don't have more valid splits for this node
                        if self.solution.tree[node_id].possible_gains[i] == 0:
                            break

                        self.solution.tree = np.copy(initial_tree)

                        self._zero_nodes(2 * node_id + 1, level + 1)
                        self._zero_nodes(2 * node_id + 2, level + 1)

                        best_split_attribute_id = int(self.solution.tree[node_id].possible_splits[i, 0])
                        best_split_threhold = self.solution.tree[node_id].possible_splits[i, 1]

                        self._split(node_id, level, best_split_attribute_id, best_split_threhold)
                        split_misclassified = self.solution.get_num_misclassified_samples()

                        # if split_misclassified < local_misclassified * self.acceptable_deterioration_factor:
                        if split_misclassified < local_misclassified:
                            local_tree = np.copy(self.solution.tree)
                            local_misclassified = split_misclassified
                            # break
                # if self.acceptable_deterioration_factor > 1:
                #     self.acceptable_deterioration_factor = self.acceptable_deterioration_factor - (self.acceptable_deterioration_factor * annealing)
                # else:
                #     self.acceptable_deterioration_factor = 1
                # if local_misclassified < initial_misclassified * self.acceptable_deterioration_factor:
                #     break

                        # print("Number of misclassifieds for attribute (%d): %d" % (i, split_misclassified))

        self.solution.tree = local_tree


    def iterated_local_search(self, time_limit):
        print("Starting standard greedy search...")

        start_time = time.time()
        self._recursive_construction(0, 0)
        greedy_misclassifieds = self.solution.get_num_misclassified_samples()
        end_time = time.time()

        print("Executed standard greedy search in %.5f seconds" % (end_time - start_time))

        if (time.time() > time_limit):
            return

        print("\n+++++++++++++++++++++++++++++++INITIAL ITERATED_SEARCH+++++++++++++++++++++++++++++++\n")

        start_time = time.time()
        annealing = 0.0001
        if greedy_misclassifieds > 0:
            print("Performing local search...")

            best_tree = np.copy(self.solution.tree)
            best_misclassifieds = self.solution.get_num_misclassified_samples()

            self._local_search(0, annealing)

            # home_base_tree = np.copy(self.solution.tree)
            # home_base_misclassifieds = self.solution.get_num_misclassified_samples()

            print("Local search finished. Misclassifieds: %s " % best_misclassifieds)
            counter = 1
            while(True):
                print("Performing perturbation...")

                minor_node = self._perturbation()

                print("Performing local search...")

                self._local_search(minor_node, annealing)
                local_misclassifieds = self.solution.get_num_misclassified_samples()

                print("Local search finished. Misclassifieds: %s " % local_misclassifieds)

                if local_misclassifieds < best_misclassifieds:
                    best_tree = np.copy(self.solution.tree)
                    best_misclassifieds = local_misclassifieds

                    print("New best_misclassifieds value.")

                # print("Acceptable deterioration factor: " + str(self.acceptable_deterioration_factor))
                # print("Counter: " + str(counter))
                # print("Home base misclassifieds with deterioration: " + str(home_base_misclassifieds * self.acceptable_deterioration_factor))
                # counter = counter + 1

                # if local_misclassifieds <= home_base_misclassifieds * self.acceptable_deterioration_factor:
                #     home_base_tree = np.copy(self.solution.tree)
                #     home_base_misclassifieds = self.solution.get_num_misclassified_samples()

                self.solution.tree = np.copy(best_tree)

                print("--------------------------")

                if (time.time() > time_limit) or (best_misclassifieds == 0):
                    break

            self.solution.tree = best_tree


        end_time = time.time()

        print("Executed iterated search in %.5f seconds" % (end_time - start_time))
        print("\n+++++++++++++++++++++++++++++++END ITERATED_SEARCH+++++++++++++++++++++++++++++++\n")
