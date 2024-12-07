import networkx as nx
import pandas as pd
from community import community_louvain
import collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from adjustText import adjust_text

# Ensure the Figures and Tables directories exist
os.makedirs('Figures', exist_ok=True)
os.makedirs('Tables', exist_ok=True)

# Load dataset
file_path = r"Data\movie_casts_sample.csv"
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
        actors = list(set(actors))
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i + 1:]:
                pair = tuple(sorted([actor1, actor2]))
                collaborations[pair] = collaborations.get(pair, 0) + 1
    
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
        movies = list(set(movies))
        for i, movie1 in enumerate(movies):
            for movie2 in movies[i + 1:]:
                pair = tuple(sorted([movie1, movie2]))
                shared_actors[pair] = shared_actors.get(pair, 0) + 1
    
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
    top_communities_ids = [comm_id for comm_id, _ in top_communities]
    community_color_map = {comm_id: color_palette(i) for i, comm_id in enumerate(top_communities_ids)}
    
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
    
    # Create labels
    texts = []
    for node in top_nodes_subgraph.nodes():
        x, y = pos[node]
        text = plt.text(x, y, node,
                        fontsize=8,
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
    
    legend_elements = []
    for comm_id, size in top_communities:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       label=f'Community {comm_id} ({size} nodes)',
                       markerfacecolor=community_color_map[comm_id],
                       markersize=10)
        )
    
    plt.legend(handles=legend_elements,
               loc='upper right',
               title='Communities',
               fontsize=10)
    
    plt.title(f"Top Nodes in Top {top_n_communities} Communities\nof {name} Network",
              pad=20,
              fontsize=18,
              fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'Figures/{name.lower()}_top_nodes_in_communities.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.3)
    plt.close()
    
    return communities, community_sizes, modularity

def compute_network_metrics(G):
    if len(G) == 0:
        return {
            'density': 0.0,
            'avg_clustering': 0.0,
            'avg_degree': 0.0,
            'min_degree': 0,
            'max_degree': 0,
            'median_degree': 0
        }
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    avg_degree = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 0
    degrees = [d for n, d in G.degree()]
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    median_degree = np.median(degrees) if degrees else 0
    return {
        'density': density,
        'avg_clustering': avg_clustering,
        'avg_degree': avg_degree,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'median_degree': median_degree
    }

def print_network_metrics(G, name):
    print(f"\n{name} Network Metrics:")
    metrics = compute_network_metrics(G)
    print(f"Network density: {metrics['density']:.4f}")
    print(f"Average clustering coefficient: {metrics['avg_clustering']:.4f}")
    print(f"Average degree: {metrics['avg_degree']:.2f}")
    print("Degree statistics:")
    print(f"  Min degree: {metrics['min_degree']}")
    print(f"  Max degree: {metrics['max_degree']}")
    print(f"  Median degree: {metrics['median_degree']:.1f}")
    
    if len(G) > 0:
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nMost central nodes:")
        for node, cent in top_nodes:
            print(f"  {node}: {cent:.4f}")

def plot_community_size_distribution(community_sizes, modularity, name):
    # Sort community sizes in descending order
    sizes = sorted(community_sizes.values(), reverse=True)
    
    n_communities = len(sizes)
    largest = max(sizes) if sizes else 0
    median_size = np.median(sizes) if sizes else 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(n_communities), sizes)
    ax.set_yscale('log')
    ax.set_xlabel("Community Rank")
    ax.set_ylabel("Community Size (log scale)")
    ax.set_title(f"{name} Network Community Size Distribution", fontsize=16, fontweight='bold')
    
    # Add a box with stats
    textstr = (f"Total Communities: {n_communities}\n"
               f"Largest Community: {largest} nodes\n"
               f"Median Size: {median_size} nodes\n"
               f"Modularity: {modularity:.4f}")
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"Figures/{name.lower()}_community_size_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


# Create and analyze networks
actor_network = create_actor_actor_network(cast_df)
movie_network = create_movie_movie_network(cast_df)

actor_communities, actor_sizes, actor_modularity = analyze_communities(actor_network, "Actor")
movie_communities, movie_sizes, movie_modularity = analyze_communities(movie_network, "Movie")

# Print network metrics
print_network_metrics(actor_network, "Actor")
print_network_metrics(movie_network, "Movie")

# Plot community size distributions
plot_community_size_distribution(actor_sizes, actor_modularity, "Actor")
plot_community_size_distribution(movie_sizes, movie_modularity, "Movie")

# Collect summary stats for LaTeX table
actor_metrics = compute_network_metrics(actor_network)
movie_metrics = compute_network_metrics(movie_network)

actor_data = {
    'Network': 'Actor',
    'Nodes': actor_network.number_of_nodes(),
    'Edges': actor_network.number_of_edges(),
    'Communities': len(set(actor_communities.values())),
    'Modularity': actor_modularity,
    'Density': actor_metrics['density'],
    'Avg Degree': actor_metrics['avg_degree'],
    'Avg Clustering': actor_metrics['avg_clustering'],
    'Max Degree': actor_metrics['max_degree']
}

movie_data = {
    'Network': 'Movie',
    'Nodes': movie_network.number_of_nodes(),
    'Edges': movie_network.number_of_edges(),
    'Communities': len(set(movie_communities.values())),
    'Modularity': movie_modularity,
    'Density': movie_metrics['density'],
    'Avg Degree': movie_metrics['avg_degree'],
    'Avg Clustering': movie_metrics['avg_clustering'],
    'Max Degree': movie_metrics['max_degree']
}

summary_df = pd.DataFrame([actor_data, movie_data])
latex_table_content = summary_df.to_latex(index=False, float_format="%.4f")

latex_table = (
r"\begin{table}[h]\centering" + "\n"
r"\caption{Summary statistics for Louvain communities of the Actor and Movie Bipartite Networks.}" + "\n"
r"\label{tab:community_summary_stats}" + "\n"
+ latex_table_content +
"\n\\end{table}"
)

with open("Tables/community_summary_stats.tex", "w") as f:
    f.write(latex_table)

print("\nAnalysis complete! Visualizations and summary stats have been saved:")
print("- Figures/actor_top_nodes_in_communities.png")
print("- Figures/movie_top_nodes_in_communities.png")
print("- Figures/actor_community_size_distribution.png")
print("- Figures/movie_community_size_distribution.png")
print("- Tables/community_summary_stats.tex")
