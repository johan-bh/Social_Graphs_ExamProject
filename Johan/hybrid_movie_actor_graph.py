import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

def plot_actor_network(cast_df, top_n=20):
    # Ensure directories exist
    os.makedirs('Figures', exist_ok=True)
    os.makedirs('Tables', exist_ok=True)

    # Get top actors by number of movies
    actor_counts = cast_df['actor_name'].value_counts().head(top_n)
    top_actors = actor_counts.index.tolist()
    
    # Create graph
    G = nx.Graph()
    
    # Filter data to relevant actors
    relevant_data = cast_df[cast_df['actor_name'].isin(top_actors)]
    
    # Add actor nodes
    for actor in top_actors:
        G.add_node(actor, node_type='actor', movie_count=actor_counts[actor])
    
    # Add movie nodes and edges from actors to movies
    movie_degrees = defaultdict(int)
    for _, row in relevant_data.iterrows():
        movie = row['movie_title']
        actor = row['actor_name']
        
        if movie not in G:
            G.add_node(movie, node_type='movie')
        G.add_edge(actor, movie)
        movie_degrees[movie] += 1
    
    # Add edges between actors who collaborated
    actor_movies = defaultdict(set)
    for _, row in relevant_data.iterrows():
        actor_movies[row['actor_name']].add(row['movie_title'])
    
    for i, actor1 in enumerate(top_actors):
        for actor2 in top_actors[i+1:]:
            shared_movies = len(actor_movies[actor1] & actor_movies[actor2])
            if shared_movies > 0:
                G.add_edge(actor1, actor2, weight=shared_movies)
    
    # Compute collaboration counts for each actor (with top actors)
    collaboration_counts = {}
    for actor in top_actors:
        collab_count = sum(
            1 for other in top_actors if other != actor and G.has_edge(actor, other)
        )
        collaboration_counts[actor] = collab_count
    
    # Convert stats to DataFrames for LaTeX output
    actor_df = pd.DataFrame({
        'Actor': top_actors,
        'Movies': [actor_counts[actor] for actor in top_actors],
        'Collaborations': [collaboration_counts[actor] for actor in top_actors]
    }).sort_values('Movies', ascending=False)
    
    actor_df_latex = actor_df.to_latex(index=False, float_format="%.0f")
    with open("Tables/top_actors_stats.tex", "w") as f:
        f.write(actor_df_latex)
    
    movie_df = pd.DataFrame({
        'Movie': list(movie_degrees.keys()),
        'TopActorsInMovie': list(movie_degrees.values())
    }).sort_values('TopActorsInMovie', ascending=False)
    
    movie_df_latex = movie_df.to_latex(index=False, float_format="%.0f")
    with open("Tables/top_movies_stats.tex", "w") as f:
        f.write(movie_df_latex)
    
    # Plot the network
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05)
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Separate nodes by type
    actor_nodes = [node for node in G.nodes() if node in top_actors]
    movie_nodes = [node for node in G.nodes() if node not in top_actors]
    
    # Node sizes
    actor_sizes = [G.degree(node) * 300 for node in actor_nodes]
    movie_sizes = [movie_degrees[node] * 200 for node in movie_nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=actor_nodes, 
                           node_color='lightblue', 
                           node_size=actor_sizes, 
                           alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, 
                           node_color='lightgreen', 
                           node_size=movie_sizes, 
                           alpha=0.5)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.5)
    
    # Labels
    actor_labels = {node: f"{node}\n({G.nodes[node]['movie_count']} movies)" for node in actor_nodes}
    movie_labels = {node: node for node in movie_nodes}
    nx.draw_networkx_labels(G, pos, actor_labels, font_size=10, font_weight='bold')
    nx.draw_networkx_labels(G, pos, movie_labels, font_size=8)
    
    # Legend handles
    actor_patch = mpatches.Patch(color='lightblue', label='Actors')
    movie_patch = mpatches.Patch(color='lightgreen', label='Movies')
    low_degree_node = Line2D([0], [0], marker='o', color='w', label='Low Degree',
                             markerfacecolor='gray', markersize=6)
    high_degree_node = Line2D([0], [0], marker='o', color='w', label='High Degree',
                              markerfacecolor='gray', markersize=15)
    
    # Bold titles
    plt.suptitle("Actor-Movie Collaboration Network", fontsize=24, y=0.97, fontweight='bold')
    plt.title("Top 20 Most Frequent Actors and Their Movie Connections", fontsize=18, y=0.94, pad=20, fontweight='bold')
    
    # Legend at top right
    legend = plt.legend(handles=[actor_patch, movie_patch, low_degree_node, high_degree_node],
                        fontsize=18, loc='upper right', bbox_to_anchor=(0.98, 1))
    
    # Collaboration stats text below the legend
    collab_text = "Top Actor Collaborations:\n"

    for actor in top_actors[:10]:
        collab_text += f"{actor}: {collaboration_counts[actor]} collaborations\n"
    
    # Add info box aligned below the legend
    plt.figtext(0.85, 0.75, collab_text, 
                fontsize=12,
                ha='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=10))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig("Figures/actor_movie_collaboration_network.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Load the dataset
    file_path = r"Data/movie_casts_sample.csv"
    cast_df = pd.read_csv(file_path)
    plot_actor_network(cast_df, top_n=20)
