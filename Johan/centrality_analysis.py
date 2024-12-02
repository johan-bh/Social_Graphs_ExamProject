import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def create_networks(cast_df):
    """Create actor-actor and movie-movie networks."""
    # Create actor-actor network
    print("Creating actor-actor network...")
    actor_network = nx.Graph()
    movie_groups = cast_df.groupby('movie_title')['actor_name'].apply(list)
    
    for movie, actors in tqdm(movie_groups.items(), desc="Processing movies"):
        actors = list(set(actors))
        for i, actor1 in enumerate(actors):
            for actor2 in actors[i + 1:]:
                if actor_network.has_edge(actor1, actor2):
                    actor_network[actor1][actor2]['weight'] += 1
                else:
                    actor_network.add_edge(actor1, actor2, weight=1)
    
    # Create movie-movie network
    print("\nCreating movie-movie network...")
    movie_network = nx.Graph()
    actor_groups = cast_df.groupby('actor_name')['movie_title'].apply(list)
    
    for actor, movies in tqdm(actor_groups.items(), desc="Processing movies"):
        movies = list(set(movies))
        for i, movie1 in enumerate(movies):
            for movie2 in movies[i + 1:]:
                if movie_network.has_edge(movie1, movie2):
                    movie_network[movie1][movie2]['weight'] += 1
                else:
                    movie_network.add_edge(movie1, movie2, weight=1)
    
    return actor_network, movie_network

def analyze_centralities(G):
    """Calculate different centrality measures."""
    print("Calculating centralities...")
    centralities = {
        'Degree': nx.degree_centrality(G),
        'Betweenness': nx.betweenness_centrality(G),
        'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
    }
    return centralities

def plot_centrality_distributions(centralities, name):
    """Plot centrality distributions with improved visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Centrality Distributions for {name} Network', y=1.02)
    
    for ax, (measure, values) in zip(axes, centralities.items()):
        # Create histogram
        ax.hist(list(values.values()), bins=50, alpha=0.7)
        
        # Add vertical line for mean
        mean_val = np.mean(list(values.values()))
        ax.axvline(mean_val, color='r', linestyle='--', alpha=0.5)
        
        # Add text with statistics
        stats_text = f'Mean: {mean_val:.3f}\nMedian: {np.median(list(values.values())):.3f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f'{measure} Centrality')
        ax.set_xlabel('Centrality Value')
        ax.set_ylabel('Frequency')
        
        # Use log scale for y-axis if distribution is highly skewed
        if np.max(list(values.values())) / np.mean(list(values.values())) > 10:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'Figures/{name.lower()}_centrality_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def print_top_nodes(centralities, name):
    """Print top nodes for each centrality measure."""
    print(f"\nTop 5 {name}s by centrality measures:")
    for measure, values in centralities.items():
        print(f"\n{measure} Centrality:")
        top_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, value in top_nodes:
            print(f"  {node}: {value:.4f}")

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    cast_df = pd.read_csv("Data/movie_casts_sample.csv")
    
    # Create networks
    actor_network, movie_network = create_networks(cast_df)
    
    # Analyze actor network
    print("\nAnalyzing actor network...")
    actor_centralities = analyze_centralities(actor_network)
    plot_centrality_distributions(actor_centralities, "Actor")
    print_top_nodes(actor_centralities, "Actor")
    
    # Analyze movie network
    print("\nAnalyzing movie network...")
    movie_centralities = analyze_centralities(movie_network)
    plot_centrality_distributions(movie_centralities, "Movie")
    print_top_nodes(movie_centralities, "Movie")
    
    print("\nAnalysis complete! Check the Figures directory for visualizations.") 