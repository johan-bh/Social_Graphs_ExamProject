import networkx as nx
import pandas as pd
from community import community_louvain
import collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from adjustText import adjust_text

# Ensure the Figures directory exists
os.makedirs('Figures', exist_ok=True)

# Load dataset
file_path = r"C:\Users\jbh\Desktop\Social_Graphs_ExamProject\Data\movie_casts_sample.csv"
cast_df = pd.read_csv(file_path)

print("Creating networks...")

# Create the actor-actor network based on shared movies
def create_actor_actor_network(cast_df):
    actor_network = nx.Graph()
    
    # Group actors by movies
    print("Creating actor-actor network...")
    movie_groups = cast_df.groupby('movie_title')['actor_name'].apply(list)
    
    # Count collaborations
    collaborations = {}
    
    for movie, actors in tqdm(movie_groups.items(), desc="Processing movies"):
        # Remove duplicates
        actors = list(set(actors))
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i + 1:]:
                pair = tuple(sorted([actor1, actor2]))
                if pair in collaborations:
                    collaborations[pair] += 1
                else:
                    collaborations[pair] = 1
    
    # Add edges with weights
    for (actor1, actor2), weight in collaborations.items():
        actor_network.add_edge(actor1, actor2, weight=weight)
    
    return actor_network

# Create the movie-movie network based on shared actors
def create_movie_movie_network(cast_df):
    movie_network = nx.Graph()
    
    # Group movies by actors
    print("Creating movie-movie network...")
    actor_groups = cast_df.groupby('actor_name')['movie_title'].apply(list)
    
    # Count shared actors
    shared_actors = {}
    
    for actor, movies in tqdm(actor_groups.items(), desc="Processing actors"):
        # Remove duplicates
        movies = list(set(movies))
        for i, movie1 in enumerate(movies):
            for movie2 in movies[i + 1:]:
                pair = tuple(sorted([movie1, movie2]))
                if pair in shared_actors:
                    shared_actors[pair] += 1
                else:
                    shared_actors[pair] = 1
    
    # Add edges with weights
    for (movie1, movie2), weight in shared_actors.items():
        movie_network.add_edge(movie1, movie2, weight=weight)
    
    return movie_network

def analyze_communities(G, name):
    print(f"\nAnalyzing {name} communities...")
    
    # Detect communities
    communities = community_louvain.best_partition(G, weight='weight')
    n_communities = len(set(communities.values()))
    community_sizes = collections.Counter(communities.values())
    
    print(f"\n{name} Community Detection Results:")
    print(f"Number of communities detected: {n_communities}")
    
    # Get top N largest communities
    top_n_communities = 5
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n_communities]
    print(f"\nTop {top_n_communities} community sizes:")
    for comm_id, size in top_communities:
        print(f"Community {comm_id}: {size} nodes")
    
    # Calculate modularity
    modularity = community_louvain.modularity(communities, G)
    print(f"Community modularity: {modularity:.4f}")
    
    # Build a subgraph consisting of top nodes from each community
    top_nodes = []
    for comm_id, _ in top_communities:
        # Nodes in the current community
        nodes_in_comm = [node for node, comm in communities.items() if comm == comm_id]
        subgraph = G.subgraph(nodes_in_comm)
        
        # Get top 5 nodes by degree within the community
        degrees = subgraph.degree()
        top_nodes_in_comm = sorted(degrees, key=lambda x: x[1], reverse=True)[:5]
        top_nodes.extend([node for node, _ in top_nodes_in_comm])
    
    # Create subgraph with top nodes and their edges
    top_nodes_subgraph = G.subgraph(top_nodes)
    
    # Assign colors to communities
    color_palette = plt.get_cmap('tab10')
    community_color_map = {}
    for idx, (comm_id, _) in enumerate(top_communities):
        community_color_map[comm_id] = color_palette(idx)
    
    # Assign colors to nodes
    node_colors = []
    for node in top_nodes_subgraph.nodes():
        comm_id = communities[node]
        node_colors.append(community_color_map[comm_id])
    
    # Create layout
    pos = nx.spring_layout(top_nodes_subgraph, seed=42)
    
    # Draw the network
    plt.figure(figsize=(15, 12))
    
    # Draw edges
    nx.draw_networkx_edges(top_nodes_subgraph, pos, alpha=0.5, width=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(top_nodes_subgraph, pos,
                          node_color=node_colors,
                          node_size=1000,
                          alpha=0.8)
    
    # Create labels with adjustText
    texts = []
    for node in top_nodes_subgraph.nodes():
        x, y = pos[node]
        text = plt.text(x, y, node,
                       fontsize=10,
                       fontweight='bold',
                       horizontalalignment='center',
                       verticalalignment='center',
                       bbox=dict(facecolor='white',
                               edgecolor='none',
                               alpha=0.7,
                               pad=2))
        texts.append(text)
    
    # Adjust label positions to avoid overlap
    adjust_text(texts,
               arrowprops=dict(arrowstyle='-',
                              color='gray',
                              lw=0.5,
                              alpha=0.5),
               expand_points=(1.5, 1.5))
    
    # Create legend
    legend_elements = []
    for comm_id, size in top_communities:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                      label=f'Community {comm_id} ({size} nodes)',
                      markerfacecolor=community_color_map[comm_id],
                      markersize=10)
        )
    
    # Position legend under the title
    plt.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.05),
              ncol=3,  # Arrange legend items in 3 columns
              title='Communities',
              fontsize=10,
              title_fontsize=12)
    
    plt.title(f"Top Nodes in Top {top_n_communities} Communities\nof {name} Network",
             pad=20,
             size=14,
             y=1.2)  # Move title up to make room for legend
    plt.axis('off')
    
    # Adjust layout to make room for title and legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the top margin
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.savefig(f'Figures/{name.lower()}_top_nodes_in_communities.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()
    
    return communities, community_sizes

def print_network_metrics(G, name):
    print(f"\n{name} Network Metrics:")
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    
    print(f"Network density: {density:.4f}")
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    print(f"Average degree: {avg_degree:.2f}")
    
    # Degree distribution statistics
    degrees = [d for n, d in G.degree()]
    print(f"Degree statistics:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Median degree: {np.median(degrees):.1f}")
    
    # Most central nodes
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nMost central nodes:")
    for node, cent in top_nodes:
        print(f"  {node}: {cent:.4f}")

# Create and analyze networks
actor_network = create_actor_actor_network(cast_df)
movie_network = create_movie_movie_network(cast_df)

# Analyze communities in both networks
actor_communities, actor_sizes = analyze_communities(actor_network, "Actor")
movie_communities, movie_sizes = analyze_communities(movie_network, "Movie")

# Print network metrics
print_network_metrics(actor_network, "Actor")
print_network_metrics(movie_network, "Movie")

print("\nAnalysis complete! Visualizations have been saved as:")
print("- Figures/actor_top_nodes_in_communities.png")
print("- Figures/movie_top_nodes_in_communities.png")
