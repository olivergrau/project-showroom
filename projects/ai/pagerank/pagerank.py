import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    all_pages = [key for key in corpus]
    target_pages = corpus[page] - set(page)

    model = dict()
    model[page] = (1 - damping_factor) * (1 / len(all_pages))

    if len(target_pages) == 0:
        for target in all_pages:
            model[target] = 1 / len(all_pages)
    else:
        for target in target_pages:
            model[target] = ((damping_factor * (1 / len(target_pages))) + 
                             ((1 - damping_factor) * (1 / len(all_pages))))

    sum_probabilities = sum(model.values())
    normalized_probabilities = {page: prob / sum_probabilities for page, prob in model.items()}
    
    return normalized_probabilities


def get_next_page(transition_probabilities):
    """    
    Chooses a key from the transition probability distribution based on the probabilities in it.
    
    Parameters:
    transition_probabilities (dict): A dictionary in which the keys are the possible states
                                     and the values represent the transition probabilities.
                                     
    Returns:
    str: The selected key based on the transition probabilities.
    """
    rand_val = random.random()

    # init cumulative probability
    cumulative_probability = 0.0
    
    for key, probability in transition_probabilities.items():
        # Add the current probability to the cumulative probability
        cumulative_probability += probability

        # If the random value is less than the cumulative probability, choose this key
        if rand_val < cumulative_probability:
            return key
        

def the_actual_sampling(corpus, damping_factor, n):
    page_counts = dict()
    all_pages = [key for key in corpus]

    for page in all_pages:
        page_counts[page] = 0

    current_page = random.choice(all_pages)

    for i in range(n):
        transitions = transition_model(corpus, current_page, damping_factor)
        next_page = get_next_page(transitions)
        page_counts[next_page] += 1
        current_page = next_page

    # normalize and create estimated probabilities    
    sum_probabilities = sum(page_counts.values())
    normalized_probabilities = {page: prob /
                                sum_probabilities for page, prob in page_counts.items()}

    return normalized_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    disjoint_sets = find_disjoint_sets(corpus)
    overall_pageranks = {}

    for subset in disjoint_sets:
        sub_corpus = {page: links.intersection(subset)
                      for page, links in corpus.items() if page in subset}
        pageranks = the_actual_sampling(sub_corpus, damping_factor, n) 
        overall_pageranks.update(pageranks)
    
    # normalize and create estimated probabilities    
    sum_probabilities = sum(overall_pageranks.values())
    normalized_probabilities = {page: prob /
                                sum_probabilities for page, prob in overall_pageranks.items()}

    return normalized_probabilities


def find_keys_with_count_links(corpus, search_key):
    result = dict()
    for key, links in corpus.items():
        if search_key in links:
            result[key] = len(links)
            
    return result


def calculate_new_pagerank(page_ranks, n, damping_factor):
    new_pagerank = (1 - damping_factor) / n
    
    sum = 0.0
    for page_rank, num_links in page_ranks.values():
        sum += page_rank / num_links
        
    return new_pagerank + (damping_factor * sum)  


def find_incoming_pageranks(corpus, page, all_pageranks):
    pageranks = dict()
    
    # find all pages in corpus that links to 'page'
    incoming_links = find_keys_with_count_links(corpus, page)

    if len(incoming_links) > 0:
        # get all page ranks for those keys from parameter page ranks
        for item, count in incoming_links.items():
            pageranks[item] = (all_pageranks[item], count)
    
    return pageranks


def find_disjoint_sets(corpus):
    """
    Finds disjoint sets of pages in the corpus.
    
    Parameters:
    corpus (dict): The Corpus Dictionary, which contains the pages and their outgoing links.
    
    Returns:
    list of sets: A list of disjoint sets of pages in the corpus.
    """
    def dfs(page, visited, component):
        stack = [page]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                stack.extend(corpus.get(node, []))
                stack.extend([key for key, values in corpus.items() if node in values])

    visited = set()
    disjoint_sets = []

    for page in corpus:
        if page not in visited:
            component = set()
            dfs(page, visited, component)
            disjoint_sets.append(component)

    return disjoint_sets


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    disjoint_sets = find_disjoint_sets(corpus)
    overall_pageranks = {}
    
    threshold = 0.001
    
    copy_corpus = corpus.copy()
    
    for page, links in copy_corpus.items():
        if len(links) == 0:
            corpus[page] = corpus.keys() | set()
    
    for subset in disjoint_sets:
        sub_corpus = {page: links.intersection(subset)
                      for page, links in corpus.items() if page in subset}
        pageranks = compute_pagerank(sub_corpus, damping_factor, threshold)
        overall_pageranks.update(pageranks)

    total_rank = sum(overall_pageranks.values())
    normalized_pageranks = {page: rank / total_rank for page, rank in overall_pageranks.items()}

    return normalized_pageranks


def compute_pagerank(corpus, damping_factor, threshold):
    pageranks = dict()
    all_pages = [key for key in corpus]

    for page in all_pages:
        pageranks[page] = 1 / len(all_pages)

    converge = dict()
    
    for page in all_pages:
        converge[page] = False

    while any(value == False for value in converge.values()):            
        for page in all_pages:
            
            if converge[page]:
                continue
                
            # find all pages that link to page
            in_pageranks = find_incoming_pageranks(corpus, page, pageranks)                                    
            new_pagerank = calculate_new_pagerank(in_pageranks, len(all_pages), damping_factor) 
            
            # check difference between new and old page rank
            dif = abs(pageranks[page] - new_pagerank)
            
            if dif <= threshold:
                converge[page] = True
                
            pageranks[page] = new_pagerank

    # normalize and create estimated probabilities    
    sum_probabilities = sum(pageranks.values())
    normalized_pageranks = {page: prob / sum_probabilities for page, prob in pageranks.items()}

    return normalized_pageranks


if __name__ == "__main__":
    main()
