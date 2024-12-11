import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

def plot_actor_movie_hybrid(cast_df, top_n=20):
    # Ensure directories exist
    os.makedirs('Figures', exist_ok=True)
    os.makedirs('Tables', exist_ok=True)

    # Get top actors by number of movies
    actor_counts = cast_df['actor_name'].value_counts().head(top_n)
    top_actors = actor_counts.index.tolist()

    # Create a bipartite graph: Actors - Movies
    B = nx.Graph()
    # Filter data to relevant actors
    relevant_data = cast_df[cast_df['actor_name'].isin(top_actors)]
    
    # Add actor nodes
    B.add_nodes_from(top_actors, bipartite=0, node_type='actor')
    
    # Add movie nodes and edges from actors to movies
    for _, row in relevant_data.iterrows():
        actor = row['actor_name']
        movie = row['movie_title']
        if not B.has_node(movie):
            B.add_node(movie, bipartite=1, node_type='movie')
        B.add_edge(actor, movie)

    # Project the bipartite graph onto actors only (to find actor-actor collaborations)
    # The actor projection creates edges between actors who share at least one movie.
    # Use a built-in projection method from networkx:
    top_actors_nodes = [n for n, d in B.nodes(data=True) if d['node_type'] == 'actor']
    A = nx.bipartite.projected_graph(B, top_actors_nodes)
    
    # Calculate shared movies (weight of edges)
    for u, v in A.edges():
        # Count how many movies these two actors share
        shared = len(set(B[u]) & set(B[v]))
        A[u][v]['weight'] = shared

    # Identify communities of actors within A
    # Using greedy modularity communities as a simple example:
    # (You can also try other algorithms if you have them available.)
    from networkx.algorithms.community import greedy_modularity_communities
    communities = greedy_modularity_communities(A, weight='weight')
    
    # Assign community IDs to each actor
    actor_community_map = {}
    for i, comm in enumerate(communities):
        for actor in comm:
            actor_community_map[actor] = i

    # Now we visualize the hybrid graph B (Actors + Movies), but color actors by their community.
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(B, k=2, iterations=50, seed=42)

    actor_nodes = [n for n, d in B.nodes(data=True) if d['node_type'] == 'actor']
    movie_nodes = [n for n, d in B.nodes(data=True) if d['node_type'] == 'movie']

    # Calculate degrees for sizing
    movie_degrees = {}
    for m in movie_nodes:
        movie_degrees[m] = B.degree(m)

    actor_sizes = [A.degree(a)*300 for a in actor_nodes]  # size by actor-actor degree in A
    movie_sizes = [movie_degrees[m]*200 for m in movie_nodes]

    # Color actors by community
    # Generate a color map for communities:
    num_communities = len(communities)
    cmap = plt.cm.get_cmap('tab20', num_communities)
    actor_colors = [cmap(actor_community_map[a]) for a in actor_nodes]

    # Draw the nodes
    nx.draw_networkx_nodes(B, pos, nodelist=actor_nodes,
                           node_color=actor_colors,
                           node_size=actor_sizes,
                           alpha=0.8, edgecolors='black')
    
    nx.draw_networkx_nodes(B, pos, nodelist=movie_nodes,
                           node_color='lightgreen',
                           node_size=movie_sizes,
                           alpha=0.5, edgecolors='gray')
    
    # Draw edges
    nx.draw_networkx_edges(B, pos, alpha=0.4, width=0.5)

    # Labels
    actor_labels = {node: f"{node}\n({actor_counts[node]} movies)" for node in actor_nodes if node in actor_counts}
    # For readability, limit labeling to the top few actors by degree or centrality, or omit movie labels entirely
    nx.draw_networkx_labels(B, pos, actor_labels, font_size=10, font_weight='bold')

    # Legend for node types
    actor_patch = mpatches.Patch(color='gray', label='Actors')
    movie_patch = mpatches.Patch(color='lightgreen', label='Movies')

    # Create a legend for communities by showing a few community colors
    community_handles = []
    for i in range(min(num_communities, 5)):
        community_handles.append(mpatches.Patch(color=cmap(i), label=f'Community {i}'))

    plt.suptitle("Actor-Movie Hybrid Graph", fontsize=24, y=0.97, fontweight='bold')
    plt.title("Highlighting Actor Clusters and Collaborative Communities", fontsize=18, y=0.94, pad=20, fontweight='bold')

    plt.legend(handles=[actor_patch, movie_patch] + community_handles,
               fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 1))

    plt.axis('off')
    # plt.tight_layout()
    # plt.savefig("Figures/actor_movie_communities.png", dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()

    # Exporting community data to a table
    community_data = []
    for i, comm in enumerate(communities):
        for actor in comm:
            community_data.append({'CommunityID': i, 'Actor': actor, 'MoviesCount': actor_counts.get(actor, 0)})

    community_df = pd.DataFrame(community_data).sort_values(['CommunityID', 'MoviesCount'], ascending=[True, False])
    community_df_latex = community_df.to_latex(index=False, float_format="%.0f")
    # with open("Tables/actor_communities.tex", "w") as f:
    #     f.write(community_df_latex)


if __name__ == "__main__":
    # Load the dataset
    file_path = r"Data/movie_casts_sample.csv"
    cast_df = pd.read_csv(file_path)
    plot_actor_movie_hybrid(cast_df, top_n=20)
