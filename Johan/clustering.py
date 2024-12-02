import networkx as nx
import pandas as pd
from community import community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load dataset
file_path = r"C:\Users\jbh\Desktop\Social_Graphs_ExamProject\Data\movie_casts_sample.csv"
cast_df = pd.read_csv(file_path)

print("Creating networks...")

# Create the actor-actor network based on shared movies
def create_actor_actor_network(cast_df):
    actor_network = nx.Graph()
    
    # Group actors by movies, using actor_name
    print("Creating actor-actor network...")
    movie_groups = cast_df.groupby('movie_title')['actor_name'].apply(list)
    
    # Count collaborations
    collaborations = {}
    
    for movie, actors in tqdm(movie_groups.items(), desc="Processing movies"):
        # Remove any duplicates in case an actor plays multiple roles
        actors = list(set(actors))
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i + 1:]:
                pair = tuple(sorted([actor1, actor2]))
                if pair in collaborations:
                    collaborations[pair] += 1
                else:
                    collaborations[pair] = 1
    
    # Add edges with weights based on number of collaborations
    for (actor1, actor2), weight in collaborations.items():
        actor_network.add_edge(actor1, actor2, weight=weight)
    
    return actor_network

# Create the movie-movie network based on shared actors
def create_movie_movie_network(cast_df):
    movie_network = nx.Graph()
    
    # Group movies by actors, using actor_name
    print("Creating movie-movie network...")
    actor_groups = cast_df.groupby('actor_name')['movie_title'].apply(list)
    
    # Count shared actors
    shared_actors = {}
    
    for actor, movies in tqdm(actor_groups.items(), desc="Processing actors"):
        # Remove any duplicates
        movies = list(set(movies))
        for i, movie1 in enumerate(movies):
            for movie2 in movies[i + 1:]:
                pair = tuple(sorted([movie1, movie2]))
                if pair in shared_actors:
                    shared_actors[pair] += 1
                else:
                    shared_actors[pair] = 1
    
    # Add edges with weights based on number of shared actors
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
    
    # Print sizes of top 5 communities
    print("\nTop 5 community sizes:")
    for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Community {comm_id}: {size} nodes")
        nodes_in_comm = [node for node, comm in communities.items() if comm == comm_id]
        print(f"Example nodes: {', '.join(nodes_in_comm[:3])}")
    
    # Calculate modularity
    modularity = community_louvain.modularity(communities, G)
    print(f"Community modularity: {modularity:.4f}")
    
    # Create two visualizations
    
    # 1. Force-directed layout for largest communities
    plt.figure(figsize=(20, 12))
    
    # Get top N largest communities
    top_n_communities = 5
    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n_communities]
    top_comm_ids = [comm_id for comm_id, _ in top_communities]
    
    # Create subgraph of only the largest communities
    nodes_in_top_comms = [node for node, comm in communities.items() if comm in top_comm_ids]
    subgraph = G.subgraph(nodes_in_top_comms)
    
    # Calculate layout for subgraph
    pos = nx.spring_layout(subgraph, k=2, iterations=100)
    
    # Create color map for top communities
    color_map = plt.cm.Set3(np.linspace(0, 1, top_n_communities))
    community_colors = dict(zip(top_comm_ids, color_map))
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5)
    
    # Draw nodes for each community
    for comm_id in top_comm_ids:
        nodes = [node for node in subgraph.nodes() if communities[node] == comm_id]
        nx.draw_networkx_nodes(subgraph, pos,
                             nodelist=nodes,
                             node_color=[community_colors[comm_id]],
                             node_size=100,
                             label=f'Community {comm_id} (size: {community_sizes[comm_id]})')
    
    # Add labels for high-degree nodes in subgraph
    degrees = dict(subgraph.degree())
    threshold = np.percentile(list(degrees.values()), 90)
    high_degree_nodes = {node: node for node, degree in degrees.items() if degree > threshold}
    nx.draw_networkx_labels(subgraph, pos, high_degree_nodes, font_size=8)
    
    plt.title(f"Top {top_n_communities} Largest Communities in {name} Network", pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_top_communities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Community size distribution plot
    plt.figure(figsize=(12, 6))
    
    # Get community sizes sorted
    sizes = sorted(community_sizes.values(), reverse=True)
    
    # Plot size distribution
    plt.bar(range(len(sizes)), sizes, alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Community Rank')
    plt.ylabel('Community Size (log scale)')
    plt.title(f'{name} Network Community Size Distribution')
    
    # Add statistics
    plt.text(0.7, 0.95, 
             f'Total Communities: {n_communities}\n'
             f'Largest Community: {max(sizes)} nodes\n'
             f'Median Size: {np.median(sizes):.1f} nodes\n'
             f'Modularity: {modularity:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_community_sizes.png', dpi=300)
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
print("- actor_network_communities.png")
print("- movie_network_communities.png")