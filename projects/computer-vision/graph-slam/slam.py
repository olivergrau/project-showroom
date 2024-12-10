import numpy as np

## slam takes in 6 arguments and returns mu, 
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations

# x = self.x + dx + self.rand() * self.motion_noise
# y = self.y + dy + self.rand() * self.motion_noise
# data[i][0] Measurements:  [[1, 0.8065670930680946, -2.6783562924892275]]
# data[i][1] Motion:  [1, 2]

def initialize_constraints(N, num_landmarks, world_size, var_motion, var_measurement):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''

    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    size_H = 2 * N + 2 * num_landmarks # square matrix    

    ## Define the constraint matrix, Omega, with two initial "strength" values
    ## for the initial x, y location of our robot (in the center)
    omega = np.zeros((size_H, size_H))

    # no information on the off diagonals while initialisation
    omega[0][0] = 1 / var_motion # diagonal entry 1
    omega[1][1] = 1 / var_motion # diagonal entry 2

    ## Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    xi = np.zeros((size_H, 1))
    xi[0] = (1 / var_measurement) * (world_size / 2) # encode the start position too
    xi[1] = (1 / var_measurement) * (world_size / 2) # encode the start position too

    return omega, xi

def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):

    def get_omega_move_update(H_size, n, variance):
        """
        Returns a matrix of the same size as H with entries populated for the movement step N.
    
        Parameters:
        - H_size: int, the size of the H matrix (number of rows or columns in H).
        - n: int, the current movement step (0-based index).
        - variance: float, the variance of the motion noise.
    
        Returns:
        - move_matrix: numpy.ndarray, a matrix of the same size as H with populated movement constraints.
        """
        # Initialize a matrix of the same size as H with zeros
        move_matrix = np.zeros((H_size, H_size))

        # Calculate the information weight as the inverse of the variance
        weight = 1 / variance

        # Determine the indices for the current movement step
        x_t = 2 * n         # x-coordinate of the current pose
        y_t = 2 * n + 1     # y-coordinate of the current pose
        x_t1 = 2 * (n + 1)  # x-coordinate of the next pose
        y_t1 = 2 * (n + 1) + 1  # y-coordinate of the next pose

        # Set the diagonal values for the current and next poses
        move_matrix[x_t, x_t] = weight
        move_matrix[y_t, y_t] = weight
        move_matrix[x_t1, x_t1] = weight
        move_matrix[y_t1, y_t1] = weight

        # Set the off-diagonal values for the coupling between current and next poses
        move_matrix[x_t, x_t1] = -weight
        move_matrix[x_t1, x_t] = -weight
        move_matrix[y_t, y_t1] = -weight
        move_matrix[y_t1, y_t] = -weight

        return move_matrix

    def get_xi_move_update(H_size, n, variance, dx, dy):
        move_vector = np.zeros((H_size, 1))

        # Calculate the information weight as the inverse of the variance
        weight = 1 / variance

        x_t = 2 * n
        y_t = 2 * n + 1
        x_t1 = 2 * (n + 1)
        y_t1 = 2 * (n + 1) + 1

        # attention here with the signs: the first xy must be negative (even if it is multiplied with negative dx/dy)
        move_vector[x_t, 0] = -weight * dx
        move_vector[y_t, 0] = -weight * dy
        move_vector[x_t1, 0] = weight * dx
        move_vector[y_t1, 0] = weight * dy

        return move_vector

    def get_omega_sense_update(H_size, n, l, variance):
        sense_update = np.zeros((H_size, H_size))

        # Calculate the information weight as the inverse of the variance
        weight = 1 / variance

        x_t = 2 * n
        y_t = 2 * n + 1
        L_x = 2 * l + (2 * N)
        L_y = 2 * l + (2 * N) + 1

        # diagonals
        sense_update[x_t, x_t] = weight
        sense_update[y_t, y_t] = weight
        sense_update[L_x, L_x] = weight
        sense_update[L_y, L_y] = weight

        # off-diagonals
        sense_update[x_t, L_x] = -weight
        sense_update[L_x, x_t] = -weight
        sense_update[y_t, L_y] = -weight
        sense_update[L_y, y_t] = -weight

        return sense_update

    def get_xi_sense_update(H_size, n, l, dx, dy, variance):
        sense_update = np.zeros((H_size, 1))

        weight = 1 / variance

        x_t = 2 * n
        y_t = 2 * n + 1
        L_x = 2 * l + (2 * N)
        L_y = 2 * l + (2 * N) + 1

        sense_update[x_t, 0] = -weight * dx
        sense_update[y_t, 0] = -weight * dy
        sense_update[L_x, 0] = weight * dx
        sense_update[L_y, 0] = weight * dy

        return sense_update

    # Let's treat the noise as std    
    var_mot = motion_noise**2
    var_meas = measurement_noise**2

    ## Use your initialization to create constraint matrices, omega and xi
    omega, xi = initialize_constraints(N, num_landmarks, world_size, var_mot, var_meas)

    ## Iterate through each time step in the data
    ## get all the motion and measurement data as you iterate

    for i in range(N-1): # when we iterate up to N then we only index the pose coordinates
        dx = data[i][1][0]
        dy = data[i][1][1]

        ## update the constraint matrix/vector to account for all *motion* and motion noise        
        omega += get_omega_move_update(omega.shape[0], i, var_mot)
        xi_upd = get_xi_move_update(omega.shape[0], i, var_mot, dx, dy)
        xi += xi_upd

        ## update the constraint matrix/vector to account for all *measurements*
        ## this should be a series of additions that take into account the measurement noise
        for measurement in data[i][0]:
            omega += get_omega_sense_update(omega.shape[0], i, measurement[0], var_meas)
            xi_upd = get_xi_sense_update(omega.shape[0], i, measurement[0], measurement[1], measurement[2], var_meas)
            xi += xi_upd

    ## After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    epsilon = 1e-6 # some stabilization
    omega += epsilon * np.eye(omega.shape[0])

    omega_inv = np.linalg.inv(np.matrix(omega))
    mu = omega_inv * xi
    #mu = np.linalg.solve(omega, xi)

    return mu # return `mu`
