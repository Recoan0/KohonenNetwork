import numpy as np
import matplotlib.pyplot as plt

# Problem might occur with subtracting vectors since they are 3 dimensional.
np.random.seed(1)  # Set seed of random generator to get same outcome each time


class Node:
    def __init__(self, weight_vector_size, x, y):
        self.weight_vector_size = weight_vector_size
        self.weight_vector = self.build_weight_vector()
        self.x = x
        self.y = y

    def build_weight_vector(self):
        return np.random.random((self.weight_vector_size, 3))  # initialize random weights (with 3 dimensions for color)

    def calculate_node_euclidian_distance(self, bmu):
        return np.sqrt(np.square(self.x - bmu.x) + np.square(self.y - bmu.y))

    def calculate_influence(self, bmu, radius):
        node_euclidian_distance = self.calculate_node_euclidian_distance(bmu)
        return np.exp(-1 * np.square(node_euclidian_distance) / (2 * np.square(radius)))

    def update_weights(self, input_vector, learning_rate, bmu, radius):
        influence = self.calculate_influence(bmu, radius)
        self.weight_vector += learning_rate * influence * (input_vector - self.weight_vector)



class KohonenNetwork:
    def __init__(self, matrix_width, matrix_height, input_vector, max_iterations):
        self.input_vector = input_vector
        self.input_vector_size = input_vector.shape[0]  # first number is length of vector
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        self.max_iterations = max_iterations

        self.node_matrix = self.create_node_matrix()  # Step 1

        self.sigma_0 = max(self.matrix_width, self.matrix_height) / 2  # Initialize radius_0, if equal to 1 will result in division by 0 error
        self.lambda_constant = max_iterations / np.log(self.sigma_0)  # Initialize time constant lambda
        self.a0 = 0.1  # Initialize learning_rate_0

    def run(self):
        current_input_vector = self.input_vector
        for t in range(self.max_iterations):  # Step 2
            bmu = self.get_bmu_node(current_input_vector)  # Step 3
            radius = self.calculate_radius(t)
            learning_rate = self.calculate_learning_rate(t)
            in_range_nodes = self.get_in_range_nodes(bmu, radius)  # Step 4
            for node in in_range_nodes:
                node.update_weights(current_input_vector, learning_rate, bmu, radius)  # Step 5
        self.visualise()

    def create_node_matrix(self):
        # Create matrix of matrix_width x matrix_height and give Nodes their coordinates in the matrix
        return np.array([[Node(self.input_vector_size, x, y) for x in range(self.matrix_width)] for y in range(self.matrix_height)])

    def calculate_radius(self, t):
        return self.sigma_0 * np.exp(-1 * (t / self.lambda_constant))  # Return calculation for sigma_t

    def calculate_learning_rate(self, t):
        return self.a0 * np.exp(-1 * (t / self.lambda_constant))  # Return calculation for alpha_t

    def get_bmu_node(self, current_input_vector):
        # Gets node with lowest euclidian distance between its weight vector and the current input vector
        distance = None
        bmu = None
        for node in self.node_matrix.ravel():
            distance_to_current_node = self.calculate_vector_euclidian_distance(current_input_vector, node.weight_vector)
            if distance is None or distance_to_current_node < distance:
                distance = distance_to_current_node
                bmu = node
        return bmu

    def get_in_range_nodes(self, bmu, radius):
        in_range_nodes = []
        for node in self.node_matrix.ravel():
            if node.calculate_node_euclidian_distance(bmu) < radius:
                in_range_nodes.append(node)
        return in_range_nodes

    def visualise(self):
        plt.plot(self.node_matrix)
        plt.show()

    @staticmethod
    def calculate_vector_euclidian_distance(input_vector, compare_vector):
        squared_distance = 0
        for i in range(input_vector.shape[0]):
            # Add square of each vector element difference
            squared_distance += np.square(input_vector[i] - compare_vector[i])
        return np.sqrt(squared_distance)


input_data = np.random.random((20, 3))
KohonenNetwork(10, 10, input_data, 100).run()
