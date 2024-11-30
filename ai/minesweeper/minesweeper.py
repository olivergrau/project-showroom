import itertools
import random
from copy import deepcopy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """
    
    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count
        self.safe_cells = set()
        self.mine_cells = set()
        
        if self.count < 0:
            raise ValueError("Count must be >= 0")

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        
        return self.mine_cells        
   
    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        
        return self.safe_cells

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        
        # check if cell is in the sentence
        if not cell in self.cells:
            return
        
        # remove cell from sentence and reduce count by 1
        self.cells.remove(cell)
        self.count = self.count - 1
        self.mine_cells.add(cell)
        
    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """

        # check if cell is in the sentence
        if not cell in self.cells:
            return
        
        # remove cell from sentence and count stays the same
        self.cells.remove(cell)        
        self.safe_cells.add(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []
        
        self.safes_with_count = set()

    def find_center_field(self, neighbors):        
        # extract lists for x- and y-coord from set of neighbors
        x_coords = [coord[0] for coord in neighbors]
        y_coords = [coord[1] for coord in neighbors]
            
        # find min and max of coords
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # calculate center of min and max coord
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
    
        # correction for edge cases
        if center_x >= self.height:
            center_x = self.height - 1
    
        if center_y >= self.width:
            center_y = self.width - 1
    
        return center_x, center_y

    def get_neighbors(self, cell):
        all_neighbors = set()
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):
                if (i, j) != (cell[0], cell[1]) and 0 <= i < self.height and 0 <= j < self.width:
                    all_neighbors.add((i, j))
        
        return all_neighbors

    def find_missing_neighbors(self, neighbors, cell):
        # Set of all possible neighbors for the field (x, y)
        all_neighbors = self.get_neighbors(cell)
    
        # Find the missing neighbors by comparing with the given set M
        missing_neighbors = all_neighbors - neighbors
    
        return missing_neighbors

    @staticmethod
    def find_subset_relationships(list_of_sets):
        subset_relationships = []
    
        # iterate over all pairs of sets
        for i in range(len(list_of_sets)):
            for j in range(i + 1, len(list_of_sets)):
                set1 = list_of_sets[i].cells
                set2 = list_of_sets[j].cells
                    
                # check if one set is a subset of another
                if set1.issubset(set2):
                    subset_relationships.append((list_of_sets[i], list_of_sets[j]))
                elif set2.issubset(set1):
                    subset_relationships.append((list_of_sets[j], list_of_sets[i]))
    
        return subset_relationships
    
    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def deduct_and_reduce(self):
        copy_of_kb = self.knowledge.copy()
        
        modified = False
        
        for sentence in copy_of_kb:
            cells_copy = sentence.cells.copy()
            if sentence.count == len(sentence.cells):             
                for mine_cell in cells_copy:
                    self.mark_mine(mine_cell)

                self.knowledge.remove(sentence)
                
                modified = True

            elif sentence.count == 0:                                
                for safe_cell in cells_copy:
                    self.mark_safe(safe_cell)

                self.knowledge.remove(sentence)
                
                modified = True
                
        if modified:
            self.deduct_and_reduce()
            
    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe, updating any sentences that contain the cell as well
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count` 
               Be sure to only include cells whose state is still undetermined in the sentence
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
               If, based on any of the sentences in self.knowledge, 
               new cells can be marked as safe or as mines, then the function should do so
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        
        # 2) mark the cell as safe, updating any sentences that contain the cell as well
        self.mark_safe(cell)
        self.deduct_and_reduce()
        self.safes_with_count.add((cell, count))
        
        # 3) add a new sentence to the AI's knowledge base
        #    based on the value of `cell` and `count` 
        #    Be sure to only include cells whose state is still undetermined in the sentence
        #    The cell cannot be a center cell already present in the knowledge 
        #
    
        cell = self.check_and_update_knowledge(cell, count)

        # ToDo: Idea: scan all safe cells again and re check (so new knowledge can be applied to already processed cells)
        for s, c in self.safes_with_count:
            if s != cell and c != count:
                self.check_and_update_knowledge(s, c)

    def check_and_update_knowledge(self, cell, count):
        
        neighbors = self.get_neighbors(cell)
        
        if count == 0:
            # means all neighbor cells are safe, update sentences in kb with that info            
            for neighbor in neighbors:
                self.mark_safe(neighbor)
        elif len(neighbors) == count:
            for neighbor in neighbors:
                self.mark_mine(neighbor)
        else:
            new_sentence = Sentence(neighbors, count)

            for mine in self.mines:
                new_sentence.mark_mine(mine)

            for safe in self.safes:
                new_sentence.mark_safe(safe)

            if len(new_sentence.cells) == new_sentence.count:
                for cell in new_sentence.cells:
                    self.mark_mine(cell)

            if new_sentence.count == 0:
                for cell in new_sentence.cells:
                    self.mark_safe(cell)

            if new_sentence.count != 0 and len(new_sentence.cells) != new_sentence.count:
                self.knowledge.append(new_sentence)
        # 4) mark any additional cells as safe or as mines
        #    if it can be concluded based on the AI's knowledge base
        #    or rephrased:
        #    If, based on any of the sentences in self.knowledge, 
        #    new cells can be marked as safe or as mines, then the function should do so
        #
        self.deduct_and_reduce()
        # 5) add any new sentences to the AI's knowledge base
        #    if they can be inferred from existing knowledge
        # More generally, any time we have two sentences set1 = count1 and set2 = count2 
        # where set1 is a subset of set2, then we can construct the new sentence set2 - set1 = count2 - count1             
        #
        if len(self.knowledge) > 1:
            subsets = self.find_subset_relationships(self.knowledge)

            while len(subsets) > 0:
                subset, superset = subsets[0]

                if len(superset.cells) == superset.count:
                    stop = True

                if len(subset.cells) == subset.count:
                    stop = True

                new_count = superset.count - subset.count
                new_set = superset.cells - subset.cells

                # replace related sentences with new sentence
                self.knowledge.remove(subset)
                self.knowledge.remove(superset)

                if new_count == 0:
                    for elem in new_set:
                        self.mark_safe(elem)
                elif new_count == len(new_set):
                    for elem in new_set:
                        self.mark_mine(elem)

                # build a new sentence
                new_sentence = Sentence(new_set, new_count)

                if new_count != 0 and new_count != len(new_set):
                    self.knowledge.append(new_sentence)

                self.deduct_and_reduce()

                # check for relations in modified knowledge            
                subsets = self.find_subset_relationships(self.knowledge)
        return cell

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        
        The move returned must be known to be safe, and not a move already made
        If no safe move can be guaranteed, the function should return None
        The function should not modify self.moves_made, self.mines, self.safes, or self.knowledge
        """
        remaining = self.safes - self.moves_made
        
        if len(remaining) > 0:
            return random.choice(tuple(remaining))
                
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if len(self.moves_made) == 0:
            return random.randint(0, self.height - 1), random.randint(0, self.width - 1)
        
        for sentence in self.knowledge:
            known_mines = sentence.known_mines()
            
            # construct a set with all available cells
            available_cells = {(i, j) for i in range(self.height) for j in range(self.width)}
            
            # subtract all mines and moves made
            remaining = available_cells - known_mines - self.moves_made

            if len(remaining) > 0:
                return random.choice(tuple(remaining))

        return None
