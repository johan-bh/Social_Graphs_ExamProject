import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from semantic_clustering import load_movie_data, MovieDialogueAnalyzer
import pandas as pd
from matplotlib.lines import Line2D
import math

def prepare_sentiment_data(movie_ids, movie_lines, sia):
    """Prepare sentiment arcs as a feature matrix for clustering."""
    sentiment_data = []
    for movie_id in movie_ids:
        movie_lines_filtered = movie_lines[movie_lines['MovieID'] == movie_id].sort_values('LineID')
        movie_lines_filtered['Sentiment'] = movie_lines_filtered['Text'].apply(
            lambda x: sia.polarity_scores(x)['compound']
        )
        movie_lines_filtered['CumulativeAverage'] = movie_lines_filtered['Sentiment'].expanding().mean()
        x_original = np.linspace(0, 1, len(movie_lines_filtered))
        y_original = movie_lines_filtered['CumulativeAverage'].values
        f_interp = interp1d(x_original, y_original, kind='linear', fill_value="extrapolate")
        x_resampled = np.linspace(0, 1, 100)
        y_resampled = f_interp(x_resampled)
        sentiment_data.append(y_resampled)
    return np.array(sentiment_data)

def create_similarity_graph(sentiment_data):
    """Create a graph with nodes as movies and edges based on cosine similarity."""
    similarity_matrix = cosine_similarity(sentiment_data)
    graph = nx.Graph()
    for i in range(len(similarity_matrix)):
        graph.add_node(i)
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity = similarity_matrix[i, j]
            if similarity > 0.5:
                graph.add_edge(i, j, weight=similarity)
    return graph

def cluster_with_louvain(graph):
    """Cluster movies using Louvain algorithm."""
    partition = community_louvain.best_partition(graph, weight='weight', resolution=1.0)
    return partition

def get_genre_frequencies(partition, movie_ids, movie_titles):
    """Calculate genre frequencies for each cluster."""
    genre_frequencies = {}
    movie_titles = movie_titles.set_index('MovieID')
    for movie_idx, cluster_id in partition.items():
        movie_id = movie_ids[movie_idx]
        genres = movie_titles.loc[movie_id, 'Genres'].split('|')
        if cluster_id not in genre_frequencies:
            genre_frequencies[cluster_id] = {}
        for genre in genres:
            genre_frequencies[cluster_id][genre] = genre_frequencies[cluster_id].get(genre, 0) + 1
    return genre_frequencies


import ast
from matplotlib import font_manager

def plot_clusters_louvain(sentiment_data, partition, movie_ids, movie_titles, genre_frequencies, enable_title=True):
    """Visualize sentiment arcs for each Louvain cluster with genre annotations and top movies.

    Parameters
    ----------
    sentiment_data : np.ndarray
        The array of sentiment arcs for each movie.
    partition : dict
        A dictionary from node index to cluster ID.
    movie_ids : list
        The list of movie IDs corresponding to each sentiment_data row.
    movie_titles : pd.DataFrame
        The movie titles DataFrame containing at least MovieID, Title, Votes, Genres.
    genre_frequencies : dict
        A dictionary with cluster_id as keys and genre frequency dictionaries as values.
    enable_title : bool, optional
        Whether to display the main figure title, by default True.
    """

    # Group movies by clusters
    clusters = {}
    for movie_idx, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(movie_idx)
    
    movie_titles_indexed = movie_titles[movie_titles['MovieID'].isin(movie_ids)].copy()
    n_clusters = len(clusters)
    
    n_cols = 1
    n_rows = math.ceil(n_clusters / n_cols)
    
    # Increase overall figure size if needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7 * n_rows))
    if n_clusters == 1:
        axes = [axes]
    
    # Font sizes
    cluster_title_fontsize = 28
    legend_fontsize = 24
    title_fontsize = 28

    # Prepare legend font properties
    legend_font_prop = font_manager.FontProperties(size=legend_fontsize, weight='bold')
    
    for cluster_idx, (cluster_id, movies) in enumerate(clusters.items()):
        ax = axes[cluster_idx]
        cluster_movies = [movie_ids[movie_idx] for movie_idx in movies]
        
        # Plot all sentiment arcs in the cluster (background individual arcs)
        for m_i, movie_idx in enumerate(movies):
            ax.plot(np.linspace(0, 1, 100), sentiment_data[movie_idx], alpha=0.3,
                    label='Individual Movies' if m_i == 0 else "")
        
        # Plot top 5 movies in the cluster
        movie_data = movie_titles_indexed[movie_titles_indexed['MovieID'].isin(cluster_movies)]
        top_movies = movie_data.nlargest(5, 'Votes')
        
        for _, movie_row in top_movies.iterrows():
            movie_id = movie_row['MovieID']
            idx = movie_ids.index(movie_id)
            ax.plot(np.linspace(0, 1, 100), sentiment_data[idx], linewidth=2,
                    label=f"Top Movie: {movie_row['Title']}")
        
        # Compute top genres and format as bullet points without strings/brackets
        genres = genre_frequencies[cluster_id]
        sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
        
        bullet_points = []
        for genre_str, count in sorted_genres[:3]:
            try:
                parsed_genres = ast.literal_eval(genre_str)
                pretty_genres = ", ".join(parsed_genres)
            except:
                pretty_genres = genre_str.strip("[]' ")
            
            bullet_points.append(f"• {pretty_genres} ({count})")
        
        top_genres_label = "Top Genres:\n" + "\n".join(bullet_points)
        
        # Remove individual subplot labels (we have global labels)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Bold and bigger cluster title
        ax.set_title(f"Cluster {cluster_idx + 1} (Movies: {len(movies)})", 
                     fontsize=cluster_title_fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Modify the legend: add top genres info as a separate (dummy) handle
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a dummy handle for top genres
        top_genres_handle = Line2D([], [], color='none')
        handles.append(top_genres_handle)
        labels.append(top_genres_label)
        
        if len(handles) > 0:
            # Remove fontsize argument and rely solely on prop
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.,
                      prop=legend_font_prop)
    
    # If more subplots than clusters, remove extra axes
    for k in range(cluster_idx + 1, len(axes)):
        axes[k].axis('off')
    
    # Add main figure title if enabled
    if enable_title:
        plt.suptitle("Sentiment Clustering of Movies Using Louvain Algorithm", 
                     fontsize=title_fontsize, y=1.02, fontweight='bold')
    
    # Adjust layout before adding fig.text labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Add global x and y labels using fig.text
    fig.text(0.2, 0.02, "Dialogue Sequence", ha='center', va='center', fontsize=28, fontweight='bold')
    fig.text(-0.002, 0.5, "Cumulative Sentiment", ha='center', va='center', rotation='vertical', fontsize=28, fontweight='bold')
    
    # Save the figure
    plt.savefig('Figures/sentiment_clusters_louvain_annotated.png', dpi=500, bbox_inches='tight')
    plt.close()



# Example Usage
if __name__ == "__main__":
    print("Loading movie dialog data...")
    movie_lines, movie_conversations, movie_titles = load_movie_data()
    analyzer = MovieDialogueAnalyzer(movie_lines, movie_conversations, movie_titles)
    
    # Use all movies in the dataset
    movie_titles['Votes'] = pd.to_numeric(movie_titles['Votes'], errors='coerce')
    all_movies = movie_titles.dropna(subset=['Votes'])
    movie_ids = all_movies['MovieID'].tolist()
    
    print("Preparing sentiment arcs...")
    sentiment_data = prepare_sentiment_data(movie_ids, movie_lines, analyzer.sia)
    
    print("Creating similarity graph...")
    similarity_graph = create_similarity_graph(sentiment_data)
    
    print("Clustering with Louvain algorithm...")
    partition = cluster_with_louvain(similarity_graph)
    
    print("Calculating genre frequencies...")
    genre_frequencies = get_genre_frequencies(partition, movie_ids, movie_titles)
    
    print("Visualizing clusters...")
    plot_clusters_louvain(sentiment_data, partition, movie_ids, movie_titles, genre_frequencies, enable_title=False)
    
    print("\nClustering and visualization complete! Check 'Figures/sentiment_clusters_louvain_annotated.png'.")
