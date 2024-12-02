import networkx as nx
import pandas as pd
from itertools import combinations

# Load the bipartite graph (constructed earlier)
file_path = r"C:\Users\jbh\Desktop\Social_Graphs_ExamProject\movie_casts_sample.csv"
movie_casts = pd.read_csv(file_path)

# Create a bipartite graph
B = nx.Graph()

# Add movie and actor nodes
movies = movie_casts['movie_title'].unique()
actors = movie_casts['actor_name'].unique()

B.add_nodes_from(movies, bipartite=0)  # Movie nodes
B.add_nodes_from(actors, bipartite=1)  # Actor nodes

# Add edges between movies and actors
for _, row in movie_casts.iterrows():
    B.add_edge(row['movie_title'], row['actor_name'])

# Actor-Actor Projection
actor_projection = nx.projected_graph(B, nodes=[n for n, d in B.nodes(data=True) if d.get("bipartite") == 1])

# Compute actor centralities
actor_degree_centrality = nx.degree_centrality(actor_projection)
actor_betweenness_centrality = nx.betweenness_centrality(actor_projection)
actor_eigenvector_centrality = nx.eigenvector_centrality(actor_projection)

# Top actors based on centralities
top_actors_degree = sorted(actor_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_actors_betweenness = sorted(actor_betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_actors_eigenvector = sorted(actor_eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 Actors by Degree Centrality:")
print(top_actors_degree)
print("\nTop 10 Actors by Betweenness Centrality:")
print(top_actors_betweenness)
print("\nTop 10 Actors by Eigenvector Centrality:")
print(top_actors_eigenvector)

# Movie-Movie Projection
movie_projection = nx.Graph()

# Create movie-movie edges based on shared actors
for movie1, movie2 in combinations(movies, 2):
    shared_actors = set(movie_casts[movie_casts['movie_title'] == movie1]['actor_name']) & \
                    set(movie_casts[movie_casts['movie_title'] == movie2]['actor_name'])
    if len(shared_actors) > 1:  # Threshold: Shared actors > 1
        movie_projection.add_edge(movie1, movie2, weight=len(shared_actors))

# Compute structural changes for movie-movie projection
movie_degrees = dict(movie_projection.degree(weight="weight"))
movie_betweenness_centrality = nx.betweenness_centrality(movie_projection, weight="weight")
movie_eigenvector_centrality = nx.eigenvector_centrality(movie_projection, weight="weight")

# Top movies based on centralities
top_movies_degree = sorted(movie_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
top_movies_betweenness = sorted(movie_betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_movies_eigenvector = sorted(movie_eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 Movies by Degree Centrality:")
print(top_movies_degree)
print("\nTop 10 Movies by Betweenness Centrality:")
print(top_movies_betweenness)
print("\nTop 10 Movies by Eigenvector Centrality:")
print(top_movies_eigenvector)
