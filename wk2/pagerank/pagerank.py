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
    probabilities = {}
    # all pages
    num_pages = len(corpus)
    # links on the current page
    num_links = len(corpus[page])
    if num_links == 0: # if no links, assume it moves randomly
        return dict.fromkeys(corpus, 1/len(corpus))

    # chance of picking a link from this page
    for link in corpus[page]:
        probabilities[link] = damping_factor / num_links
    # 1 - damping factor pick a random page entirely
    for p in corpus:
        probabilities[p] = probabilities.get(p, 0) + (1 - damping_factor) / num_pages

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # pick initial page randomly
    current_page = random.choice(list(corpus.keys()))
    proportions = dict.fromkeys(corpus.keys(), 0)
    for i in range(n):
        model = transition_model(corpus, current_page, damping_factor)
        # then we can iterate through the keys and values of the dict and when the sum of all the values is greater than the num, we pick that page?
        r = random.random()
        total = 0
        for key, value in model.items():
            if r >= total and r <= total + value:
                current_page = key
                break
            total += value
        # update proportions
        proportions[current_page] = proportions.get(current_page, 0) + 1/n
    return proportions


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold = 0.001
    N = len(corpus)
    # assign each page rank 1/n
    ranks = dict.fromkeys(corpus, 1/N)
        
    while True:
        # page rank(p) = (1-d) / N + d (sum of (PR(i) / NumLinks(i)))
        new_ranks = {}
        for p in corpus:
            total = 0
            # add incoming page ranks
            for i in corpus:
                # if empty page, add rank equally
                if len(corpus[i]) == 0:
                    total += ranks[i] / N
                # if our page is linked, add the rank / total links on that page
                elif p in corpus[i]:
                    total += ranks[i] / len(corpus[i])
            new_ranks[p] = (1 - damping_factor) / N + damping_factor * total
        if all(abs(ranks[p] - new_ranks[p]) < threshold for p in corpus):
            return new_ranks
        ranks = new_ranks


if __name__ == "__main__":
    main()
