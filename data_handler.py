import numpy as np
import time

class Instance:

    def __init__(self, instance_path, solution_path, seed, max_depth, max_time):
        self.instance_path = instance_path
        self.solution_path = solution_path
        self.seed = seed
        self.max_depth = max_depth
        self.max_time = max_time

        self._load_dataset(instance_path)

    def _load_dataset(self, instance_path):
        start = time.time()
        with open(instance_path) as fp:
            self.dataset_name = fp.readline().split()[1]
            self.num_samples = int(fp.readline().split()[1])
            self.num_attributes = int(fp.readline().split()[1])
            self.attribute_types = fp.readline().split()[1:]
            if not set(self.attribute_types).issubset({'N', 'C'}):
                raise ValueError("ERROR: non recognized attribute type")
            self.num_classes = int(fp.readline().split()[1])
            self.data_attributes = np.zeros((self.num_samples, self.num_attributes), dtype=np.float)
            self.data_classes = np.zeros(self.num_samples, dtype=np.int)

            is_finished = False
            for i, line in enumerate(fp):
                if line.strip() == "EOF":
                    is_finished = True
                    break
                values = line.split()
                self.data_attributes[i] = np.array(values[:-1]).astype(np.float)
                self.data_classes[i] = int(values[-1])
            self.num_levels = self.data_attributes.max(axis=0).astype(np.int) + 1
            if not is_finished:
                raise IOError("ERROR when reading instance, EOF has not been found where expected")
            if self.data_classes.max() >= self.num_classes:
                raise ValueError("ERROR: class indices should be in 0...num_classes-1")
        delta = time.time() - start
        print("----- DATASET [" + self.dataset_name + "] LOADED IN " + str(delta) + " (s)")
        print("----- NUMBER OF SAMPLES: " + str(self.num_samples))
        print("----- NUMBER OF ATTRIBUTES: " + str(self.num_attributes))
        print("----- NUMBER OF CLASSES: " + str(self.num_classes))
