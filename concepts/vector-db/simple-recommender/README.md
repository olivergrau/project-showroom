# Using a Vector Database to Recommend Movies

Vector search is critical for generative AI, but also have lots of interesting applications outside of gen AI. 
One very common one is building personalized recommendations. 

In this exercise, we'll take a small diversion and build a quick movie recommender using a vector database.

For this recommender, we'll use the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/), 
which contains 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. 
The strategy we'll use is to create embeddings for the movies based on the overall watch history of users. 
Then if a user watches a particular movie, we'll recommend "similar" movies, 
where similar is determined by revealed preferences, 
rather than by pre-determined features like genre, release year, etc.


## Loading data

We'll end up using the `ratings.csv` as our data table and we'll also use `movies.csv` and `links.csv` to get the movie metadata.

Let's start by reading in the `ratings.csv` file. We'll use this to compute the content embeddings.


## Computing ratings

Now we'll use the ratings data to create a new reviews dataframe of users (index) and movies (columns). 
Each entry (i, j) in the dataframe will be the rating that user_i gave to movie_j. 
If no such pair exists, then fill in the value 0.

**HINT** In Excel this would be called a pivot table


## Computing embeddings

Now let's use [matrix factorization](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf) to extract content embeddings.

We'll write a function to compute the content embeddings from the reviewmatrix dataframe and name the result `embeddings`.

**HINT**
1. SVD is a popular matrix factorization technique
2. If you're not sure which of the SVD results to use as the content embeddings, look at the shape of the results


## Metadata

Read in the `movies.csv` and `links.csv` files and make sure it is aligned with the embeddings dataframe.

**HINT** pandas provides `reindex` functionality to help with data alignment


## Create vector database table

Now that we have all of the data, let's create a table with the following fields:

1. an integer movie id field
2. a vector field of embeddings
3. a string field of genres
4. a string field for the movie title
5. an integer field for the imdb_id

First, we'll create a pydantic model named `Content` for these fields, and prepare a list of python dicts with all of the data.
Then, like we've done before, we'll connect to the local database at ~/.lancedb
and create the LanceDB table named "movielens_small".

## Generating recommendations

Finally we're ready to generate recommendations based on content vector similarity.

For this exercise please fill in the rest of the function to generate recommendations

**HINT** It's easier if you use the pydantic integration to convert results
