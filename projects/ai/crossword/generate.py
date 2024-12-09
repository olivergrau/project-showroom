import sys
from copy import deepcopy
from itertools import combinations

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v, domain in self.domains.items():
            for s in set(domain):  
                if len(s) != v.length:
                    domain.remove(s)

    @staticmethod
    def filter_words(set_a, set_b, index_a, index_b):        
        words_to_keep = set()
    
        for word_a in set_a:            
            has_match = any(
                word_b for word_b in set_b
                if len(word_a) > index_a and len(word_b) > index_b and word_a[index_a] == word_b[index_b]
            )
            if has_match:
                words_to_keep.add(word_a)
        
        # only change referenced set A
        set_a.intersection_update(words_to_keep)
    
    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        if (x, y) not in self.crossword.overlaps:
            return False
        
        overlap = self.crossword.overlaps[(x, y)]        
        len_domain_x = len(self.domains[x])                                        
        self.filter_words(self.domains[x], self.domains[y], overlap[0], overlap[1])
        
        if len(self.domains[x]) != len_domain_x:
            return True
    
        return False

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """        
        if arcs is None:
            arcs = [arc for arc, overlap in self.crossword.overlaps.items() if overlap is not None]
                
        while len(arcs) > 0:
            arc = arcs.pop()            
                
            if self.revise(arc[0], arc[1]):                
                if len(self.domains[arc[0]]) == 0:
                    return False
                
                neighbors = self.crossword.neighbors(arc[0])
                for z in neighbors - {arc[1]}:
                    arcs.append((z, arc[0]))
            
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) != len(self.crossword.variables):
            return False
        
        return True

    @staticmethod
    def find_matching_pairs(assignments, overlaps):
        result = {}

        pairs = combinations(assignments.keys(), 2)
    
        for key1, key2 in pairs:
            if (key1, key2) in overlaps:
                if overlaps[(key1, key2)] is not None:
                    result[(key1, key2)] = overlaps[(key1, key2)]
            elif (key2, key1) in overlaps:
                if overlaps[(key2, key1)] is not None:
                    result[(key2, key1)] = overlaps[(key2, key1)]
    
        return result

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        
        # distinct words check
        word_set = set(assignment.values())
        
        if len(word_set) != len(assignment):
            return False

        # check length constraint for each item
        for v, word in assignment.items():            
            if len(word) != v.length:
                return False

        # check the overlaps
        to_check = self.find_matching_pairs(assignment, self.crossword.overlaps)
        
        for (v1, v2), overlap in to_check.items():
            word1 = assignment[v1]
            word2 = assignment[v2]
            
            if word1[overlap[0]] != word2[overlap[1]]:
                return False
            
        return True    

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        
        # only work on a copy we won't change anything here
        domain_values = self.domains[var]
        
        # get neighbors
        neighbors = self.crossword.neighbors(var)

        # remove nodes from assignment
        to_consider = neighbors - assignment.keys()

        affected_nodes = dict()
        
        # Note that any variable present in assignment already has a value, 
        # and therefore shouldn’t be counted when computing the number of values 
        # ruled out for neighboring unassigned variables.
        
        for domain_value in domain_values:
            nodes = 0
            
            # take the neighbors and look in how many the same value is in the domain set
            for neighbor_node in to_consider:                
                neighbor_domain = self.domains[neighbor_node]
                
                # are there values of neighboring node that can be ruled out?
                # is there a conflict between the value of x and a value of neighbor_node?
                for word in neighbor_domain:
                    overlap = self.crossword.overlaps[(var, neighbor_node)]                    
                
                    if overlap is not None:                        
                        if domain_value[overlap[0]] != word[overlap[1]]:
                            nodes += 1
                    
            affected_nodes[domain_value] = nodes        
        
        return sorted(affected_nodes, key=lambda k: affected_nodes[k])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = self.crossword.variables - assignment.keys()

        unassigned = {key: value for key, value in self.domains.items() if key in unassigned}
        min_length = min(len(value) for value in unassigned.values())
        min_keys = [key for key, value in self.domains.items() if len(
            value) == min_length and key in unassigned]
        
        if len(min_keys) == 1:
            return min_keys[0]
        
        # apply max degree heuristic (which node has the most neighbors)
        neighbors = {key: self.crossword.neighbors(key) for key in min_keys}
        max_length = max(len(value) for value in neighbors.values())
        max_keys = [key for key, value in neighbors.items() if len(value) == max_length]
        
        return max_keys[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        
        var = self.select_unassigned_variable(assignment)
        
        for value in self.order_domain_values(var, assignment):
            copy_assignment = assignment.copy()
            copy_assignment[var] = value
            if self.consistent(copy_assignment):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
                
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
