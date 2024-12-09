from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
# Do not confuse A with AKnight, A is here a placeholder which can be AKnight or AKnave
knowledge0 = And(
    # It is either a Knight or a Knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    
    # The Knight says: I am both a knight and a knave.
    Implication(AKnight, And(AKnight, AKnave)),
    
    # The Knave says: I am both a knight and a knave.
    Implication(AKnave, Not(And(AKnight, AKnave))) 
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
# Do not confuse A with AKnight, A is here a placeholder which can be AKnight or AKnave
knowledge1 = And(
    # Figure A: It is either a Knight or a Knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    
    # Figure B: It is either a Knight or a Knave
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    
    # Figure A: The Knight says "We are both knaves."
    Implication(AKnight, And(AKnave, BKnave)),

    # Figure A: The Knave says "We are both knaves."
    Implication(AKnave, Not(And(AKnave, BKnave))),
    
    # Figure B: BKnight and BKnave say nothing
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # Figure A: It is either a Knight or a Knave
    And(
        Or(AKnight, AKnave), 
        Not(And(AKnight, AKnave))
    ),

    # Figure B: It is either a Knight or a Knave
    And(
        Or(BKnight, BKnave), 
        Not(And(BKnight, BKnave))
    ),

    # Figure A: The Knight says "We are the same kind."
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),

    # Figure A: The Knave says "We are the same kind."
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),

    # Figure B: The Knight says "We are of different kinds."
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),

    # Figure B: The Knave says "We are of different kinds."
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Figure A: It is either a Knight or a Knave
    And(
        Or(AKnight, AKnave),
        Not(And(AKnight, AKnave))
    ),

    # Figure B: It is either a Knight or a Knave
    And(
        Or(BKnight, BKnave),
        Not(And(BKnight, BKnave))
    ),

    # Figure C: It is either a Knight or a Knave
    And(
        Or(CKnight, CKnave),
        Not(And(CKnight, CKnave))
    ),

    # Figure A: The Knight says "I am a knight." or "I am a knave.", but you don't know which.
    Implication(AKnight, And(Or(AKnight, AKnave), Not(And(AKnight, AKnave)))),

    # Figure A: The Knave says "I am a knight." or "I am a knave.", but you don't know which.
    Implication(AKnave, Not(And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))))),

    # Figure B: The Knight says "A said 'I am a knave'."
    Implication(BKnight, And(
        Or(
            Implication(AKnight, BKnave), Implication(AKnave, BKnave)), Not(And(Implication(AKnight, BKnave), Implication(AKnave, BKnave))))),
        
    # Figure B: The Knave says "A said 'I am a knave'."
    Implication(BKnave, Not(And(
        Or(
            Implication(AKnight, BKnave), Implication(AKnave, BKnave)),
        Not(And(Implication(AKnight, BKnave), Implication(AKnave, BKnave)))))),    

    # Figure B: The Knight then says "C is a knave."
    Implication(BKnight, CKnave),

    # Figure B: The Knave then says "C is a knave."
    Implication(BKnave, Not(CKnave)),

    # Figure C: The Knight says "A is a knight."
    Implication(CKnight, AKnight),

    # Figure C: The Knave says "A is a knight."
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()