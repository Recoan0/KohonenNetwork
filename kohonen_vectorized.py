import numpy as np
import matplotlib.pyplot as plt
import time

# Improve by using numpy parallel functions, masks and matrix (element wise or not) multiplications to update weights

np.random.seed(4)  # Set seed of random generator to get same outcome each time


class Node:
    def __init__(self, weight_vector_size, x, y):
        self.weight_vector_size = weight_vector_size
        self.weight_vector = self.build_weight_vector()
        self.x = x
        self.y = y

    def build_weight_vector(self):
        return np.random.random(self.weight_vector_size)  # initialize random weights

    def calculate_node_euclidian_distance(self, bmu):
        return np.sqrt(np.square(self.x - bmu.x) + np.square(self.y - bmu.y))

    def calculate_influence(self, bmu, radius):
        node_euclidian_distance = self.calculate_node_euclidian_distance(bmu)
        return np.exp(-1 * np.square(node_euclidian_distance) / (2 * np.square(radius)))

    def update_weights(self, input_vector, learning_rate, bmu, radius):
        influence = self.calculate_influence(bmu, radius)
        self.weight_vector += learning_rate * influence * (input_vector - self.weight_vector)


class KohonenNetwork:
    def __init__(self, matrix_width, matrix_height, training_data, max_iterations):
        self.training_data = training_data
        self.input_vector_size = training_data.shape[1]  # first number is length of vector
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        self.max_iterations = max_iterations

        self.node_matrix = self.create_node_matrix()  # Step 1

        self.sigma_0 = max(self.matrix_width, self.matrix_height) / 2  # Initialize radius_0, if equal to 1 will result in division by 0 error
        self.lambda_constant = max_iterations / np.log(self.sigma_0)  # Initialize time constant lambda
        self.a0 = 0.1  # Initialize learning_rate_0

    def run(self):
        training_data_index = 0  # To enumerate through training data instances
        for t in range(self.max_iterations):  # Step 2
            current_input_vector = self.training_data[training_data_index]
            bmu = self.get_bmu_node(current_input_vector)  # Step 3
            radius = self.calculate_radius(t)
            learning_rate = self.calculate_learning_rate(t)
            in_range_nodes = self.get_in_range_nodes(bmu, radius)  # Step 4
            for node in in_range_nodes:
                node.update_weights(current_input_vector, learning_rate, bmu, radius)  # Step 5
            training_data_index = (training_data_index + 1) % self.training_data.shape[0]  # Wrap around if necessary
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
        plt.imshow(self.build_color_matrix(self.node_matrix))
        plt.show()

    @staticmethod
    def build_color_matrix(node_matrix):
        # Create color matrix of node-matrix shape with 3 dimensional colors
        color_matrix = np.zeros(node_matrix.shape + (3,))
        for x in range(node_matrix.shape[0]):
            for y in range(node_matrix.shape[1]):
                color_matrix[x][y] = node_matrix[x][y].weight_vector
        return color_matrix

    @staticmethod
    def calculate_vector_euclidian_distance(input_vector, compare_vector):
        return np.squeeze(np.sqrt(np.sum(np.square(input_vector - compare_vector))))


begin_time = time.time()
training_data = np.random.random((20, 3))
network = KohonenNetwork(100, 100, training_data, 2000)
network.visualise()  # visualise begin state
network.run()  # Run Kohonen algorithm and show result
print(time.time() - begin_time)
