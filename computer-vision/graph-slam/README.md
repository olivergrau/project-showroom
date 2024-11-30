# Landmark Detection & Robot Tracking (SLAM)

## Project Overview

This project involves implementing SLAM (Simultaneous Localization and Mapping) in a 2D environment. SLAM enables a robot to track its location in real-time while simultaneously mapping the locations of landmarks, such as buildings, trees, or rocks, based only on sensor and motion data. This is a crucial capability in robotics and autonomous systems, allowing robots to navigate and interact with their environment effectively.

The system demonstrates how a robot can construct a map of its surroundings and identify landmarks using measurements of movement and sensor data over time. The project focuses on the implementation and exploration of SLAM concepts in a simulated 2D world.

![2D Robot World with Landmarks](./images/robot_world.png)

## Key Components

1. **Robot Movement and Sensing**: Explore how a robot senses its environment and moves within a 2D grid world.
2. **Constraint Representation**: Understand and implement the mathematical foundations of SLAM, including constraint matrices (`omega` and `xi`).
3. **Landmark Detection and Mapping**: Detect landmarks using sensor measurements and update the map as the robot moves.
4. **SLAM Algorithm**: Implement the SLAM algorithm to estimate both the robot's trajectory and the locations of landmarks in the environment.

## Demonstrations and Applications

- **Real-Time Localization**: Track the robot’s position in the world based on noisy sensor and motion data.
- **Landmark Mapping**: Identify and map the positions of fixed landmarks in the robot's environment.
- **Mathematical Foundations**: Apply constraint-solving techniques to estimate positions and refine the map over time.
- **Simulation and Visualization**: Simulate a robot navigating a 2D grid and visualize the resulting map and trajectory.

This project showcases the intersection of robotics, mathematical optimization, and computer vision to solve the challenging problem of simultaneous localization and mapping.
