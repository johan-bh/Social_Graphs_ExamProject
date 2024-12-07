import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from adjustText import adjust_text  # Import adjustText
import os

def create_and_visualize_bipartite_network(cast_df, min_actor_movies=5, min_movie_actors=5, output_filename=None):
    """Create and visualize a bipartite network of actors and movies."""
    print(f"Creating bipartite network for min_actor_movies={min_actor_movies}, min_movie_actors={min_movie_actors}...")
    
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
    
    B.add_nodes_from(actors, bipartite=0, node_type='actor')
    B.add_nodes_from(movies, bipartite=1, node_type='movie')
    
    # Add edges
    edges = list(zip(filtered_df['actor_name'], filtered_df['movie_title']))
    B.add_edges_from(edges)
    
    print(f"Network created with {len(actors)} actors and {len(movies)} movies, and {len(B.edges())} edges.")
    
    # Only visualize if we have a non-empty network
    if len(actors) == 0 or len(movies) == 0:
        print("No nodes meet the criteria. No visualization created.")
        return B
    
    print("Calculating layout...")
    pos = nx.spring_layout(B, k=0.3, iterations=50, seed=42)
    
    # Calculate node sizes based on degree
    actor_degrees = dict(B.degree(actors))
    movie_degrees = dict(B.degree(movies))
    
    # Exponential scaling for node sizes
    actor_sizes = [50 + (actor_degrees[node] ** 2) * 2 for node in actors]
    movie_sizes = [50 + (movie_degrees[node] ** 2) * 2 for node in movies]
    
    # Create visualization with adjusted figure size
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Draw edges
    nx.draw_networkx_edges(B, pos,
                           edge_color='gray',
                           alpha=0.2,
                           width=1)
    
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
    
    # Label top actors and movies
    top_actors = sorted(actor_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_movies = sorted(movie_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    actor_labels = {node: f"Actor: {node}" for node, degree in top_actors}
    movie_labels = {node: f"Movie: {node}" for node, degree in top_movies}
    
    texts = []
    for node, label in actor_labels.items():
        x, y = pos[node]
        text = plt.text(x, y, label,
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        texts.append(text)
    for node, label in movie_labels.items():
        x, y = pos[node]
        text = plt.text(x, y, label,
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        texts.append(text)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Add a large title (suptitle) and a smaller subtitle (title)
    plt.suptitle("Actor-Movie Bipartite Network", fontsize=22, fontweight='bold')
    plt.title(f"(min_actor_movies={min_actor_movies}, min_movie_actors={min_movie_actors})\nNode sizes represent the number of connections", 
              fontsize=14)
    
    # Adjust legend to be bigger
    size_legend = [
        Line2D([0], [0], marker='o', color='w', label='Low Degree',
               markerfacecolor='gray', markersize=5),
        Line2D([0], [0], marker='o', color='w', label='High Degree',
               markerfacecolor='gray', markersize=15)
    ]
    
    actor_patch = mpatches.Patch(color='#ff7f0e', label='Actors')
    movie_patch = mpatches.Patch(color='#1f77b4', label='Movies')
    legend_elements = [actor_patch, movie_patch] + size_legend
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
               fontsize=12, title='Node Types and Sizes', title_fontsize=14)
    
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for suptitle
    
    # Save the plot if output filename is provided
    if output_filename:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        print(f"Saving visualization to {output_filename}...")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return B

if __name__ == "__main__":
    # Load the dataset
    print("Loading data...")
    file_path = r"Data/movie_casts_sample.csv"
    cast_df = pd.read_csv(file_path)
    
    print("\nInitial distribution:")
    print(f"Total actors: {len(cast_df['actor_name'].unique())}")
    print(f"Total movies: {len(cast_df['movie_title'].unique())}")
    
    # Create a folder for tables
    os.makedirs("Tables", exist_ok=True)
    
    # Prepare a DataFrame to collect summary stats
    summary_data = []

    # Visualize networks for min_actor_movies=2,3,4 (keeping min_movie_actors fixed, e.g., =2)
    for min_collab in [2, 3, 4]:
        B = create_and_visualize_bipartite_network(
            cast_df, 
            min_actor_movies=min_collab, 
            min_movie_actors=2,
            output_filename=f'Figures/actor_movie_network_min{min_collab}.png'
        )
        
        # Collect summary stats
        num_actors = len([n for n, d in B.nodes(data=True) if d.get('node_type') == 'actor'])
        num_movies = len([n for n, d in B.nodes(data=True) if d.get('node_type') == 'movie'])
        num_edges = B.number_of_edges()
        density = nx.density(B)
        
        summary_data.append({
            'min_actor_movies': min_collab,
            'min_movie_actors': 2,
            'num_actors': num_actors,
            'num_movies': num_movies,
            'num_edges': num_edges,
            'density': density
        })
        
        # Print network metrics for this configuration
        print(f"\nNetwork Metrics (min_actor_movies={min_collab}, min_movie_actors=2):")
        if len(B) > 0:
            print(f"Network density: {density:.4f}")
            
            # Get top 5 nodes by degree
            degrees = dict(B.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print("Top 5 most connected nodes:")
            for node, degree in top_nodes:
                node_type = 'Actor' if B.nodes[node]['node_type'] == 'actor' else 'Movie'
                print(f"{node_type}: {node} (Connections: {degree})")
        else:
            print("No network to report metrics on.")
    
    # Create a DataFrame and save as LaTeX table
    summary_df = pd.DataFrame(summary_data)
    latex_table = summary_df.to_latex(index=False, float_format="%.4f")
    
    with open("Tables/summary_stats.tex", "w") as f:
        f.write(latex_table)
    
    print("Summary statistics LaTeX table saved to Tables/summary_stats.tex")
