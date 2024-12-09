import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# NONTERMINALS = """
# S -> N V
# """

NONTERMINALS = """
S -> CS | NP VP | NP VP NP | NP VP PP | VP NP
CS -> S Conj S
VP -> AdvP V | V | V NP | V NP PP | V Adv
PP -> P NP | P NP Adv
NP -> AdjP N | N | Det N | Det N PP | P Det AdjP N | Det AdjP N | P N
AdjP -> Adj AdjP | Adj
AdvP -> Adv AdvP | Adv
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # parse it with word_tokenize(sentence)
    tokens = nltk.word_tokenize(sentence)
    
    # filter out non-alphabetic tokens and lower all words, so we don't have to care about capital letters
    return [token.lower() for token in tokens if token.isalpha()]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    def contains_label(subtree, label):
        return any(child.label() == label for child in subtree.subtrees() if child != subtree)

    subtrees = [subtree for subtree in tree.subtrees(
        filter=lambda t: t.label() == "NP" and not contains_label(t, "NP"))]
        
    return [subtree for subtree in subtrees]


if __name__ == "__main__":
    main()
