# Solve Sudoku with AI

## Synopsis

This project extends a Sudoku-solving agent to handle diagonal Sudoku puzzles and implements an advanced constraint-solving strategy called "naked twins." 

A diagonal Sudoku puzzle follows the same rules as traditional Sudoku but includes an additional constraint: the two main diagonals of the board must also contain the digits 1-9, without repetition. The "naked twins" strategy enhances the solving process by identifying pairs of boxes in a unit that share the same two possible digits, allowing these digits to be eliminated as possibilities from all other boxes in the unit.

This project demonstrates how AI techniques, such as constraint propagation and search algorithms, can be combined to solve complex logic puzzles effectively. It also highlights the flexibility of these methods in adapting to new constraints, such as diagonal rules in Sudoku.

## Visualization

The project includes an optional visualization tool that uses the `pygame` library to display the step-by-step progress of the Sudoku solver, making it easier to observe how the constraints and strategies work together to find a solution.
