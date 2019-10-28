import numpy as np
import matplotlib.pyplot as plt
import time

# Improved by using numpy parallel functions, masks and matrix (element wise or not) multiplications to update weights
# 100x100 for 1000 iterations takes 1.311 seconds
# 100x100 for 2000 iterations takes 2.107 seconds

np.random.seed(4)  # Set seed of random generator to get same outcome each time


class KohonenNetwork:
    @staticmethod
    def run(matrix_width, matrix_height, training_data, max_iterations):
        matrix = KohonenNetwork.create_matrix(matrix_height, matrix_width)
        sigma_0 = max(matrix_width, matrix_height) / 2  # Initialize radius_0, if equal to 1 will result in division by 0 error
        lambda_constant = max_iterations / np.log(sigma_0)  # Initialize time constant lambda
        a0 = 0.1  # Initialize learning_rate_0

        training_data_index = 0  # To enumerate through training data instances
        for t in range(max_iterations):  # Step 2
            current_input_vector = training_data[training_data_index]
            bmu_location = KohonenNetwork.get_bmu_location(matrix, current_input_vector)  # Step 3
            radius = KohonenNetwork.calculate_radius(sigma_0, lambda_constant, t)
            learning_rate = KohonenNetwork.calculate_learning_rate(a0, lambda_constant, t)
            in_range_mask = KohonenNetwork.get_in_range_mask(matrix, bmu_location, radius)  # Step 4
            matrix = KohonenNetwork.update_weights(matrix, in_range_mask, current_input_vector, learning_rate, bmu_location, radius)  # Step 5
            training_data_index = (training_data_index + 1) % training_data.shape[0]  # Wrap around if necessary
        KohonenNetwork.visualise(matrix)

    @staticmethod
    def create_matrix(height, width):
        # Create matrix of matrix_height x matrix_width (rows x columns) x 3 (rgb)
        return np.random.random((height, width, 3))

    @staticmethod
    def calculate_radius(sigma_0, lambda_constant, t):
        return sigma_0 * np.exp(-1 * (t / lambda_constant))  # Return calculation for sigma_t

    @staticmethod
    def calculate_learning_rate(a0, lambda_constant, t):
        return a0 * np.exp(-1 * (t / lambda_constant))  # Return calculation for alpha_t

    @staticmethod
    def get_bmu_location(matrix, current_input_vector):
        # Gets location of node with lowest euclidian distance between its weight vector and the current input vector
        vector_distance_matrix = KohonenNetwork.calculate_vector_euclidian_distance(current_input_vector, matrix)
        return KohonenNetwork.get_min_location(vector_distance_matrix)

    @staticmethod
    def calculate_vector_euclidian_distance(input_vector, compare_vector):
        return np.sqrt(np.sum(np.square(input_vector - compare_vector), axis=2))

    @staticmethod
    def get_min_location(matrix):
        return np.unravel_index(np.argmin(matrix), matrix.shape)

    @staticmethod
    def get_in_range_mask(matrix, bmu_location, radius):
        location_matrix = KohonenNetwork.get_location_matrix(matrix)
        distance_matrix = KohonenNetwork.calculate_node_euclidian_distance(bmu_location, location_matrix)
        in_range_mask = np.less(distance_matrix, radius)
        in_range_mask[bmu_location] = False  # We dont update the BMU itself
        return np.resize(in_range_mask, in_range_mask.shape + (1,))

    @staticmethod
    def get_location_matrix(matrix):
        return np.moveaxis(np.indices((matrix.shape[0], matrix.shape[1])), 0, -1)

    @staticmethod
    def calculate_node_euclidian_distance(bmu_location, location_matrix):
        return np.sqrt(np.sum(np.square((location_matrix - bmu_location)), axis=2))

    @staticmethod
    def calculate_influence(matrix, bmu_location, radius):
        euclidian_distances = KohonenNetwork.calculate_node_euclidian_distance(bmu_location, KohonenNetwork.get_location_matrix(matrix))
        influences = np.exp(-1 * np.square(euclidian_distances) / (2 * np.square(radius)))
        return np.resize(influences, influences.shape + (1,))

    @staticmethod
    def update_weights(matrix, in_range_mask, input_vector, learning_rate, bmu_location, radius):
        influences = KohonenNetwork.calculate_influence(matrix, bmu_location, radius)
        return matrix + in_range_mask * learning_rate * influences * (input_vector - matrix)  # In range mask makes updates 0 for nodes not in range

    @staticmethod
    def visualise(matrix):
        plt.imshow(matrix)
        plt.show()


begin_time = time.time()
network_training_data = np.random.random((20, 3))
# network.visualise()  # visualise begin state
KohonenNetwork.run(100, 100, network_training_data, 1000)  # Run Kohonen algorithm and show result
print(time.time() - begin_time)
