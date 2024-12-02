import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from adjustText import adjust_text  # Import adjustText

def create_and_visualize_bipartite_network(cast_df, min_actor_movies=5, min_movie_actors=5):
    """Create and visualize a bipartite network of actors and movies."""
    print("Creating bipartite network...")
    
    # Filter data
    actor_counts = cast_df['actor_name'].value_counts()
    movie_counts = cast_df['movie_title'].value_counts()
    
    # Filtering criteria
    qualified_actors = actor_counts[actor_counts >= min_actor_movies].index
    qualified_movies = movie_counts[movie_counts >= min_movie_actors].index
    
    filtered_df = cast_df[
        cast_df['actor_name'].isin(qualified_actors) & 
        cast_df['movie_title'].isin(qualified_movies)
    ]
    
    # Create bipartite graph
    B = nx.Graph()
    
    # Add nodes with bipartite attribute
    actors = set(filtered_df['actor_name'])
    movies = set(filtered_df['movie_title'])
    
    # Add nodes
    B.add_nodes_from(actors, bipartite=0, node_type='actor')
    B.add_nodes_from(movies, bipartite=1, node_type='movie')
    
    # Add edges
    edges = list(zip(filtered_df['actor_name'], filtered_df['movie_title']))
    B.add_edges_from(edges)
    
    print(f"Network created with {len(actors)} actors and {len(movies)} movies")
    
    # Calculate node positions using spring layout
    print("Calculating layout...")
    pos = nx.spring_layout(B, k=0.3, iterations=50, seed=42)
    
    # Calculate node sizes based on degree
    actor_degrees = dict(B.degree(actors))
    movie_degrees = dict(B.degree(movies))
    
    # Increase discrepancy between small and big nodes
    # Exponential scaling
    actor_sizes = [50 + (actor_degrees[node] ** 2) * 2 for node in actors]
    movie_sizes = [50 + (movie_degrees[node] ** 2) * 2 for node in movies]
    
    # Create visualization with adjusted figure size
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Draw edges with increased alpha
    nx.draw_networkx_edges(B, pos,
                           edge_color='gray',
                           alpha=0.2,  # Increased alpha for edges
                           width=1)    # Increased width for edges
    
    # Draw actor nodes
    nx.draw_networkx_nodes(B, pos,
                           nodelist=actors,
                           node_color='#ff7f0e',  # Orange
                           node_size=actor_sizes,
                           alpha=0.6,
                           label='Actors')
    
    # Draw movie nodes
    nx.draw_networkx_nodes(B, pos,
                           nodelist=movies,
                           node_color='#1f77b4',  # Blue
                           node_size=movie_sizes,
                           alpha=0.6,
                           label='Movies')
    
    # Add labels for top 10 actors and movies
    # Get top 10 actors and movies by degree
    top_actors = sorted(actor_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_movies = sorted(movie_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Prepare labels with differentiation
    actor_labels = {node: f"Actor: {node}" for node, degree in top_actors}
    movie_labels = {node: f"Movie: {node}" for node, degree in top_movies}
    
    # Use adjustText to avoid label overlapping
    texts = []
    # Add actor labels
    for node, label in actor_labels.items():
        x, y = pos[node]
        text = plt.text(x, y, label,
                        fontsize=8,
                        fontweight='bold',
                        color='black',  # Changed from fontcolor to color
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        texts.append(text)
    # Add movie labels
    for node, label in movie_labels.items():
        x, y = pos[node]
        text = plt.text(x, y, label,
                        fontsize=8,
                        fontweight='bold',
                        color='black',  # Changed from fontcolor to color
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        texts.append(text)
    
    # Adjust labels to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Update the title to include a descriptive subtitle
    plt.title("Actor-Movie Bipartite Network\nNode sizes represent the number of connections",
              fontsize=18, fontweight='bold')
    
    # Create custom legend
    # Add node size legend entries
    size_legend = [
        Line2D([0], [0], marker='o', color='w', label='Low Degree',
               markerfacecolor='gray', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='High Degree',
               markerfacecolor='gray', markersize=15)
    ]
    
    # Add node color legend entries
    actor_patch = mpatches.Patch(color='#ff7f0e', label='Actors')
    movie_patch = mpatches.Patch(color='#1f77b4', label='Movies')
    
    # Combine legends
    legend_elements = [actor_patch, movie_patch] + size_legend
    
    # Move legend to the top right
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=10, title='Node Types and Sizes')
    
    # Remove axes
    plt.axis('off')
    
    # Save the plot
    print("Saving visualization...")
    plt.tight_layout()
    plt.savefig('Figures/actor_movie_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return B


if __name__ == "__main__":
    # Load the dataset
    print("Loading data...")
    file_path = r"Data/movie_casts_sample.csv"
    cast_df = pd.read_csv(file_path)
    
    # Print initial distribution
    print("\nInitial distribution:")
    print(f"Total actors: {len(cast_df['actor_name'].unique())}")
    print(f"Total movies: {len(cast_df['movie_title'].unique())}")
    
    # Create and visualize the network with adjusted thresholds
    B = create_and_visualize_bipartite_network(cast_df, 
                                             min_actor_movies=5, 
                                             min_movie_actors=2)
    
    # Print some network metrics
    print("\nNetwork Metrics:")
    print(f"Network density: {nx.density(B):.4f}")
    
    # Get top actors and movies by degree
    degrees = dict(B.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 10 most connected nodes:")
    for node, degree in top_nodes:
        node_type = 'Actor' if B.nodes[node]['node_type'] == 'actor' else 'Movie'
        print(f"{node_type}: {node} (Connections: {degree})")
