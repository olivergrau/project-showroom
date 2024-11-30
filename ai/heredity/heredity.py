import csv
import itertools
import sys
import copy

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):                
        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def calculate_father_mother_children(father, mother, kids, already_considered):
    probability = 1
    
    for kid in kids:        
        # calculate individual probability mother (this only works for top level parents!!!)        
        father_p = PROBS["gene"][father["gene"]] * PROBS["trait"][father["gene"]][father["trait"]]
        
        # calculate individual probability father (this only works for top level parents!!!)        
        mother_p = PROBS["gene"][mother["gene"]] * PROBS["trait"][mother["gene"]][mother["trait"]]
        
        if not mother["name"] in already_considered.keys():
            probability *= mother_p
            already_considered[mother["name"]] = mother_p

        if not father["name"] in already_considered.keys():
            probability *= father_p
            already_considered[father["name"]] = father_p

        # calculate kid
        kid_p = (calculate_child_genes_prob(mother["gene"], father["gene"], kid["gene"])
                 * PROBS["trait"][kid["gene"]][kid["trait"]])

        if not kid["name"] in already_considered.keys():
            probability *= kid_p
            already_considered[kid["name"]] = kid_p

    return probability


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # deep copy people so we don't change it
    copy_people = copy.deepcopy(people)
    
    additional_info = {
        "gene": 0,
        "trait": False
    }
    
    # enrich with additional info
    copy_people = {key: {**value, **additional_info} for key, value in copy_people.items()}
    
    # apply given information by parameters
    for person, props in copy_people.items():
        if person in one_gene:
            props["gene"] = 1

        if person in two_genes:
            props["gene"] = 2

        if person in have_trait:
            props["trait"] = True

    grouped = reorganize_family_data(copy_people)
    
    already_considered = dict()
    probability = 1
    for record in grouped:
        probability *= calculate_father_mother_children(
            record["father"], record["mother"], record["children"], already_considered)
    
    return probability


def reorganize_family_data(people):
    def find_person_by_name(name, person_list):
        for person in person_list:
            if person['name'] == name:
                return person
        return None
    
    def build_key(father, mother):
        return father + "-" + mother
    
    parents = []
    kids = []

    for person, data in people.items():
        if any(child['mother'] == data['name'] or child['father'] == data['name'] for child in people.values()):
            parents.append(data)
        
        if data['mother'] is not None or data['father'] is not None:
            kids.append(data)

    families = []
    already_considered = set()
    map = dict()
    
    for kid in kids:
        # get the parents
        father = find_person_by_name(kid["father"], parents)
        mother = find_person_by_name(kid["mother"], parents)
        
        # put the parents into families if not already in it
        if father["name"] not in already_considered and mother["name"] not in already_considered:
            record = {"father": father, "mother": mother, "children": [kid]}
            families.append(record)
            
            already_considered.add(father["name"])
            already_considered.add(mother["name"])
            
            map[build_key(father["name"], mother["name"])] = record
        else:
            record = map[build_key(father["name"], mother["name"])]    
            
            # then add the kid to the children key
            record["children"].append(kid)
        
    return families


def calculate_child_genes_prob(mother_gene, father_gene, child_genes):

    mutation_prob = PROBS["mutation"]

    if mother_gene == 2:
        p_gene_from_mother = 1 - mutation_prob
    elif mother_gene == 1:
        p_gene_from_mother = 0.5
    else:
        p_gene_from_mother = mutation_prob

    if father_gene == 2:
        p_gene_from_father = 1 - mutation_prob
    elif father_gene == 1:
        p_gene_from_father = 0.5
    else:
        p_gene_from_father = mutation_prob

    if child_genes == 2:
        return p_gene_from_mother * p_gene_from_father
    elif child_genes == 1:
        return (p_gene_from_mother * (1 - p_gene_from_father)) + ((1 - p_gene_from_mother) * p_gene_from_father)
    else:
        return (1 - p_gene_from_mother) * (1 - p_gene_from_father)


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person, dist in probabilities.items():
        # update gene distribution
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
    
        # update trait distribution
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    
    for person, dist in probabilities.items():
        # gene distribution
        normalized_gene = {index: prob / sum(dist["gene"].values())
                           for index, prob in dist["gene"].items()}
        probabilities[person]["gene"] = normalized_gene
    
        # trait distribution   
        normalized_trait = {index: prob / sum(dist["trait"].values())
                            for index, prob in dist["trait"].items()}
        probabilities[person]["trait"] = normalized_trait


if __name__ == "__main__":
    main()
