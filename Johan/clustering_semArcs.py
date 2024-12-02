import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from semantic_clustering import load_movie_data, MovieDialogueAnalyzer
import pandas as pd

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
    partition = community_louvain.best_partition(graph, weight='weight', resolution=2.0)
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

def plot_clusters_louvain(sentiment_data, partition, movie_ids, movie_titles, genre_frequencies):
    """Visualize sentiment arcs for each Louvain cluster with genre annotations and top movies."""
    clusters = {}
    for movie_idx, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(movie_idx)
    
    # Create a copy of movie_titles and filter
    movie_titles_indexed = movie_titles.copy()
    movie_titles_indexed = movie_titles_indexed[movie_titles_indexed['MovieID'].isin(movie_ids)]
    
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Parameters for subplot organization
    max_clusters_per_figure = 6
    n_figures = (len(clusters) + max_clusters_per_figure - 1) // max_clusters_per_figure
    
    for fig_num in range(n_figures):
        # Get clusters for this figure
        start_idx = fig_num * max_clusters_per_figure
        end_idx = min((fig_num + 1) * max_clusters_per_figure, len(sorted_clusters))
        current_clusters = sorted_clusters[start_idx:end_idx]
        
        # Create figure
        fig, axes = plt.subplots(len(current_clusters), 1, figsize=(15, 5 * len(current_clusters)))
        if len(current_clusters) == 1:
            axes = [axes]
        
        for subplot_idx, (cluster_id, movies) in enumerate(current_clusters):
            ax = axes[subplot_idx]
            cluster_movies = [movie_ids[movie_idx] for movie_idx in movies]
            
            # Plot all sentiment arcs in the cluster
            for movie_idx in movies:
                ax.plot(np.linspace(0, 1, 100), sentiment_data[movie_idx], alpha=0.3, 
                       label='Individual Movies' if movie_idx == movies[0] else "")
            
            # Plot top 5 movies in the cluster
            movie_data = movie_titles_indexed[movie_titles_indexed['MovieID'].isin(cluster_movies)]
            top_movies = movie_data.nlargest(5, 'Votes')
            
            for _, movie_row in top_movies.iterrows():
                movie_id = movie_row['MovieID']
                idx = movie_ids.index(movie_id)
                ax.plot(np.linspace(0, 1, 100), sentiment_data[idx], linewidth=2, 
                       label=f"Top Movie: {movie_row['Title']}")
            
            # Add genre annotations
            genres = genre_frequencies[cluster_id]
            sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
            top_genres = ', '.join([f"{genre} ({count})" for genre, count in sorted_genres[:3]])
            ax.text(0.98, 0.9, f"Top Genres: {top_genres}", 
                    transform=ax.transAxes, fontsize=10, ha='right', 
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title(f"Cluster {cluster_id} (Movies: {len(movies)})")
            ax.set_xlabel("Normalized Time")
            ax.set_ylabel("Cumulative Sentiment")
            ax.grid(True, alpha=0.3)
            
            # Add legend for this subplot
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend(loc='upper left', fontsize=8)
        
        # Add a main title
        plt.suptitle(f"Sentiment Clustering of Movies (Part {fig_num + 1}/{n_figures})", 
                    fontsize=16, y=1.02)
        
        # Add a caption
        plt.figtext(0.5, 0.01, 
                   "Sentiment arcs grouped by Louvain clustering. "
                   "Top 5 movies highlighted with genre frequencies.", 
                   wrap=True, horizontalalignment='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f'Figures/sentiment_clusters_louvain_part{fig_num+1}.png', 
                    dpi=300, bbox_inches='tight')
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
    plot_clusters_louvain(sentiment_data, partition, movie_ids, movie_titles, genre_frequencies)
    
    print("\nClustering and visualization complete! Check 'Figures/sentiment_clusters_louvain_annotated.png'.")
