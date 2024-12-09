import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import kagglehub
import ast
import nltk
from scipy.stats import mannwhitneyu, skew, kurtosis
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
import string
from adjustText import adjust_text
import community.community_louvain as community_louvain  # For Louvain clustering
import ast

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)

def load_movie_data():
    """Load and preprocess the Cornell Movie Dialog Corpus."""
    print("Loading movie dialog dataset...")
    path = kagglehub.dataset_download("rajathmc/cornell-moviedialog-corpus")
    
    # Load movie lines
    movie_lines = pd.read_csv(
        f"{path}/movie_lines.txt", 
        sep=r' \+\+\+\$\+\+\+ ',
        header=None,
        on_bad_lines='skip',
        encoding='ISO-8859-1',
        engine='python'
    )
    movie_lines.columns = ['LineID', 'CharacterID', 'MovieID', 'CharacterName', 'Text']
    
    # Load movie conversations
    movie_conversations = pd.read_csv(
        f"{path}/movie_conversations.txt", 
        sep=r' \+\+\+\$\+\+\+ ',
        header=None,
        on_bad_lines='skip',
        encoding='ISO-8859-1',
        engine='python'
    )
    movie_conversations.columns = ['CharacterID1', 'CharacterID2', 'MovieID', 'Conversation']
    
    # Load movie titles and metadata
    movie_titles = pd.read_csv(
        f"{path}/movie_titles_metadata.txt",
        sep=r' \+\+\+\$\+\+\+ ',
        header=None,
        encoding='ISO-8859-1',
        engine='python'
    )
    movie_titles.columns = ['MovieID', 'Title', 'Year', 'Rating', 'Votes', 'Genres']
    
    # Clean the data
    movie_lines['CharacterID'] = movie_lines['CharacterID'].str.strip()
    movie_conversations['CharacterID1'] = movie_conversations['CharacterID1'].str.strip()
    movie_conversations['CharacterID2'] = movie_conversations['CharacterID2'].str.strip()
    
    # Convert IDs to string
    movie_lines['CharacterID'] = movie_lines['CharacterID'].astype(str)
    movie_conversations['CharacterID1'] = movie_conversations['CharacterID1'].astype(str)
    movie_conversations['CharacterID2'] = movie_conversations['CharacterID2'].astype(str)
    
    # Fill NaN values
    movie_lines['Text'] = movie_lines['Text'].fillna('')
    
    # Create a dictionary for faster line lookup
    line_dict = dict(zip(movie_lines['LineID'], movie_lines['Text']))
    
    def parse_conversation(conv_str):
        try:
            conv_str = conv_str.strip("[]' ").replace("'", "").replace('"', '')
            return [x.strip() for x in conv_str.split(',') if x.strip()]
        except:
            return []
    
    def get_conversation_text(conv_list):
        texts = []
        for line_id in conv_list:
            text = line_dict.get(line_id.strip(), '')
            if text:  # Only add non-empty texts
                texts.append(text)
        return ' '.join(texts)
    
    # Process conversations
    print("Processing conversations...")
    movie_conversations['Conversation'] = movie_conversations['Conversation'].apply(parse_conversation)
    movie_conversations['Text'] = movie_conversations['Conversation'].apply(get_conversation_text)
    
    # Remove empty conversations
    movie_conversations = movie_conversations[movie_conversations['Text'].str.len() > 0]
    
    print(f"Loaded {len(movie_lines)} lines from {len(movie_titles)} movies")
    print(f"Found {len(movie_conversations)} valid conversations")
    
    return movie_lines, movie_conversations, movie_titles

class MovieDialogueAnalyzer:
    def __init__(self, movie_lines, movie_conversations, movie_titles):
        self.movie_lines = movie_lines
        self.movie_conversations = movie_conversations
        self.movie_titles = movie_titles
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuations = set(string.punctuation)
        
    def create_interaction_network(self, movie_id):
        """Create a directed interaction network for a specific movie with sentiment on edges."""
        movie_convos = self.movie_conversations[self.movie_conversations['MovieID'] == movie_id]
        movie_lines_subset = self.movie_lines[self.movie_lines['MovieID'] == movie_id]
        char_names = dict(zip(movie_lines_subset['CharacterID'], movie_lines_subset['CharacterName']))
        
        MG = nx.DiGraph()
        
        line_to_character = dict(zip(movie_lines_subset['LineID'], movie_lines_subset['CharacterID']))
        line_dict = dict(zip(movie_lines_subset['LineID'], movie_lines_subset['Text']))
        
        for _, row in movie_convos.iterrows():
            try:
                conversation_ids = row['Conversation']
                for i in range(len(conversation_ids) - 1):
                    line_id_1 = conversation_ids[i]
                    line_id_2 = conversation_ids[i + 1]
                    
                    char1 = line_to_character.get(line_id_1)
                    char2 = line_to_character.get(line_id_2)
                    
                    text = line_dict.get(line_id_1, '')
                    sentiment = self.sia.polarity_scores(text)['compound']
                    
                    if char1 and char2:
                        if MG.has_edge(char1, char2):
                            old_sentiment = MG[char1][char2]['sentiment']
                            old_weight = MG[char1][char2]['weight']
                            new_sentiment = (old_sentiment * old_weight + sentiment) / (old_weight + 1)
                            MG[char1][char2]['sentiment'] = new_sentiment
                            MG[char1][char2]['weight'] += 1
                        else:
                            MG.add_edge(char1, char2, weight=1, sentiment=sentiment)
                if char1:
                    MG.add_node(char1, name=char_names.get(char1, char1))
                if char2:
                    MG.add_node(char2, name=char_names.get(char2, char2))
            except:
                continue
        
        # Remove self-loops
        MG.remove_edges_from(nx.selfloop_edges(MG))
        
        return MG, char_names

    def plot_combined_sentiment_arcs(self, top_ids, bottom_ids):
        """Plot cumulative sentiment arcs for top 10 and bottom 10 in one figure 
        with fill areas depending on sign of cumulative sentiment."""
        n_cols = 10
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 8))
        axes = axes.flatten()
        
        # Plot top 10 in the first row
        for i, movie_id in enumerate(top_ids):
            ax = axes[i]
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == movie_id].sort_values('LineID')
            movie_lines['Sentiment'] = movie_lines['Text'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
            movie_lines['CumulativeAverage'] = movie_lines['Sentiment'].expanding().mean()
            
            x = np.arange(len(movie_lines))
            y = movie_lines['CumulativeAverage'].values
            
            ax.plot(x, y, color='black', linestyle='-', linewidth=2)
            
            # Fill where y >= 0 (positive sentiment)
            ax.fill_between(x, y, where=(y >= 0), interpolate=True, color='green', alpha=0.3)
            # Fill where y < 0 (negative sentiment)
            ax.fill_between(x, y, where=(y < 0), interpolate=True, color='red', alpha=0.3)
            
            # Remove individual titles and labels
            ax.set_title("")     # no subplot title
            ax.set_xlabel("")    # no individual x-label
            ax.set_ylabel("")    # no individual y-label
            ax.grid(True, alpha=0.3)
        
        # Plot bottom 10 in the second row
        start_idx = len(top_ids)  # Usually 10
        for j, movie_id in enumerate(bottom_ids):
            ax = axes[start_idx + j]
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == movie_id].sort_values('LineID')
            movie_lines['Sentiment'] = movie_lines['Text'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
            movie_lines['CumulativeAverage'] = movie_lines['Sentiment'].expanding().mean()
            
            x = np.arange(len(movie_lines))
            y = movie_lines['CumulativeAverage'].values
            
            ax.plot(x, y, color='black', linestyle='-', linewidth=2)
            
            # Fill where y >= 0 (positive sentiment)
            ax.fill_between(x, y, where=(y >= 0), interpolate=True, color='green', alpha=0.3)
            # Fill where y < 0 (negative sentiment)
            ax.fill_between(x, y, where=(y < 0), interpolate=True, color='red', alpha=0.3)
            
            # Remove individual titles and labels
            ax.set_title("")     # no subplot title
            ax.set_xlabel("")    # no individual x-label
            ax.set_ylabel("")    # no individual y-label
            ax.grid(True, alpha=0.3)
        
        # Add a single x-axis and y-axis label for the entire figure

        # fig.supxlabel("Dialogue Sequence", fontsize=13, fontweight='bold')
        # fig.supylabel("Cumulative Sentiment", fontsize=13, fontweight='bold')

        # fig.subplots_adjust(left=0.2)  # Increase the left margin
        # fig.subplots_adjust(bottom=0.2)  # Increase the bottom margin
        # fig.subplots_adjust(left=0.1)  # Increase the left margin
        fig.text(0.5, -0.02, "Dialogue Sequence", ha='center', va='center', fontsize=32, fontweight='bold')
        fig.text(-0.005, 0.5, "Cumulative Sentiment", ha='center', va='center', rotation='vertical', fontsize=32, fontweight='bold')


        plt.tight_layout()
        plt.savefig('Figures/sentiment_arcs_top_bottom.png', dpi=600, bbox_inches='tight')
        plt.close()



    def plot_combined_interaction_networks(self, top_ids, bottom_ids, title=False):
        """Plot interaction networks for top 5 and bottom 5 in a 5x2 grid.
        
        Left column: top 5 movies (one per row)
        Right column: bottom 5 movies (one per row)
        """
        n_rows = 5
        n_cols = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 30))
        if title:
            fig.suptitle(
                "Character Interaction Networks\nTop 5 Movies (Left Column) vs Bottom 5 Movies (Right Column)\nEdge colors indicate sentiment (blue=negative, red=positive)",
                fontsize=20, fontweight='bold', y=1.02
            )
        
        # Plot top movies in the left column
        for i, movie_id in enumerate(top_ids):
            ax = axes[i, 0]
            movie = self.movie_titles[self.movie_titles['MovieID'] == movie_id].iloc[0]
            
            MG, char_names = self.create_interaction_network(movie_id)
            
            # Transform sentiments
            for u, v, data in MG.edges(data=True):
                transformed_weight = (data['sentiment'] + 1) / 2.0
                data['weight'] = transformed_weight
            
            UG = MG.to_undirected()
            partition = community_louvain.best_partition(UG, weight='weight')
            
            communities = set(partition.values())
            community_palette = sns.color_palette("hls", len(communities))
            community_color_map = {c: community_palette[i_c] for i_c, c in enumerate(communities)}
            node_colors = [community_color_map[partition[node]] for node in UG.nodes()]
            
            pos = nx.kamada_kawai_layout(MG)
            edge_colors = [MG[u][v]['sentiment'] for u, v in MG.edges()]
            
            nx.draw_networkx_nodes(MG, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(MG, pos, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm,
                                edge_vmin=-1, edge_vmax=1, width=2, alpha=1.0,
                                arrows=False, ax=ax)
            
            labels = {node: char_names.get(node, str(node)) for node in MG.nodes()}
            # Make node labels bold by setting font_weight='bold'
            nx.draw_networkx_labels(MG, pos, labels=labels, font_size=16, font_weight='bold', ax=ax,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            ax.set_title(f"{movie['Title'].title()}\n{MG.number_of_nodes()} chars, {MG.number_of_edges()} interactions",
                        fontsize=24, fontweight='bold')
            ax.axis('off')
        
        # Plot bottom movies in the right column
        for i, movie_id in enumerate(bottom_ids):
            ax = axes[i, 1]
            movie = self.movie_titles[self.movie_titles['MovieID'] == movie_id].iloc[0]
            
            MG, char_names = self.create_interaction_network(movie_id)
            
            for u, v, data in MG.edges(data=True):
                transformed_weight = (data['sentiment'] + 1) / 2.0
                data['weight'] = transformed_weight
            
            UG = MG.to_undirected()
            partition = community_louvain.best_partition(UG, weight='weight')
            
            communities = set(partition.values())
            community_palette = sns.color_palette("hls", len(communities))
            community_color_map = {c: community_palette[i_c] for i_c, c in enumerate(communities)}
            node_colors = [community_color_map[partition[node]] for node in UG.nodes()]
            
            pos = nx.kamada_kawai_layout(MG)
            edge_colors = [MG[u][v]['sentiment'] for u, v in MG.edges()]
            
            nx.draw_networkx_nodes(MG, pos, node_color=node_colors, node_size=400, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(MG, pos, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm,
                                edge_vmin=-1, edge_vmax=1, width=2, alpha=1.0,
                                arrows=False, ax=ax)
            
            labels = {node: char_names.get(node, str(node)) for node in MG.nodes()}
            # Make node labels bold here as well
            nx.draw_networkx_labels(MG, pos, labels=labels, font_size=16, font_weight='bold', ax=ax,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            ax.set_title(f"{movie['Title'].title()}\n{MG.number_of_nodes()} chars, {MG.number_of_edges()} interactions",
                        fontsize=24, fontweight='bold')
            ax.axis('off')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        # cbar.set_label('Edge Sentiment Score', fontsize=20,  fontweight='bold')
        
        # Add global y-axis label (if desired)
        fig.text(0.94, 0.5, "Edge Sentiment Score", ha='center', va='center', fontsize=22, fontweight='bold', rotation='vertical')
        
        plt.tight_layout(rect=[0, 0, 0.93, 0.92])  # Decrease top margin from 1.0 to 0.95

        plt.savefig('Figures/interaction_networks_top_bottom.png', dpi=600, bbox_inches='tight')
        plt.close()



    def plot_combined_emotional_keywords(self, top_ids, bottom_ids):
        """Plot emotional keywords for top 10 and bottom 10 in one figure."""
        n_cols = 10
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12))
        fig.suptitle('Emotional Keywords\nTop 10 Movies (Top Row) vs Bottom 10 Movies (Bottom Row)', 
                     fontsize=22, fontweight='bold', y=1.05)
        
        # Top row
        for i, movie_id in enumerate(top_ids):
            ax = axes[0, i]
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == movie_id]
            text = ' '.join(movie_lines['Text'].tolist())
            
            words = nltk.word_tokenize(text)
            words = [word.lower() for word in words if word.isalpha()]
            words = [word for word in words if word not in self.stop_words and word not in self.punctuations]
            
            word_freq = Counter(words)
            
            word_sentiments = {}
            for word, freq in word_freq.items():
                sentiment = self.sia.polarity_scores(word)['compound']
                if abs(sentiment) > 0.1:
                    word_sentiments[word] = {'sentiment': sentiment, 'frequency': freq}
            
            if word_sentiments:
                sorted_words = sorted(
                    word_sentiments.items(),
                    key=lambda x: (abs(x[1]['sentiment']), x[1]['frequency']),
                    reverse=True
                )[:10]
                
                words_plot = [w for w, data in sorted_words]
                sentiments = [data['sentiment'] for w, data in sorted_words]
                colors = ['red' if s < 0 else 'green' for s in sentiments]
                ax.barh(words_plot, sentiments, color=colors, alpha=0.6)
                ax.set_xlim(-1, 1)
            else:
                ax.text(0.5, 0.5, 'No Significant Keywords Found', ha='center', va='center')
            
            title = self.movie_titles[self.movie_titles['MovieID'] == movie_id]['Title'].iloc[0]
            ax.set_title(title.title(), fontsize=10, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Keywords')
        
        # Bottom row
        for i, movie_id in enumerate(bottom_ids):
            ax = axes[1, i]
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == movie_id]
            text = ' '.join(movie_lines['Text'].tolist())
            
            words = nltk.word_tokenize(text)
            words = [word.lower() for word in words if word.isalpha()]
            words = [word for word in words if word not in self.stop_words and word not in self.punctuations]
            
            word_freq = Counter(words)
            
            word_sentiments = {}
            for word, freq in word_freq.items():
                sentiment = self.sia.polarity_scores(word)['compound']
                if abs(sentiment) > 0.1:
                    word_sentiments[word] = {'sentiment': sentiment, 'frequency': freq}
            
            if word_sentiments:
                sorted_words = sorted(
                    word_sentiments.items(),
                    key=lambda x: (abs(x[1]['sentiment']), x[1]['frequency']),
                    reverse=True
                )[:10]
                
                words_plot = [w for w, data in sorted_words]
                sentiments = [data['sentiment'] for w, data in sorted_words]
                colors = ['red' if s < 0 else 'green' for s in sentiments]
                ax.barh(words_plot, sentiments, color=colors, alpha=0.6)
                ax.set_xlim(-1, 1)
            else:
                ax.text(0.5, 0.5, 'No Significant Keywords Found', ha='center', va='center')
            
            title = self.movie_titles[self.movie_titles['MovieID'] == movie_id]['Title'].iloc[0]
            ax.set_title(title.title(), fontsize=10, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Keywords')
        
        plt.tight_layout(rect=[0, 0, 1.0, 1.0])
        plt.savefig('Figures/emotional_keywords_top_bottom.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compute_advanced_network_metrics(self, movie_ids):
        records = []
        
        for movie_id in movie_ids:
            MG, char_names = self.create_interaction_network(movie_id)
            
            UG = MG.to_undirected()
            
            n_nodes = MG.number_of_nodes()
            n_edges = MG.number_of_edges()
            density = nx.density(UG)
            
            degrees = [d for n, d in MG.degree()]
            avg_degree = np.mean(degrees) if degrees else 0.0
            
            if n_nodes > 1:
                avg_clustering = nx.average_clustering(UG)
            else:
                avg_clustering = 0.0
            
            for u, v, data in MG.edges(data=True):
                data['weight_transformed'] = (data['sentiment'] + 1) / 2.0
            partition = community_louvain.best_partition(UG, weight='weight_transformed')
            modularity = community_louvain.modularity(partition, UG, weight='weight_transformed')
            
            edge_sentiments = [data['sentiment'] for u, v, data in MG.edges(data=True)]
            mean_sentiment = np.mean(edge_sentiments) if edge_sentiments else 0.0
            std_sentiment = np.std(edge_sentiments) if edge_sentiments else 0.0
            skew_sentiment = skew(edge_sentiments) if len(edge_sentiments) > 1 else 0.0
            kurt_sentiment = kurtosis(edge_sentiments) if len(edge_sentiments) > 1 else 0.0
            
            if n_nodes > 0:
                components = nx.connected_components(UG)
                largest_comp = max(components, key=len)
                LCC = UG.subgraph(largest_comp).copy()
                
                if LCC.number_of_nodes() > 1:
                    try:
                        avg_shortest_path = nx.average_shortest_path_length(LCC)
                    except:
                        avg_shortest_path = np.nan
                    try:
                        diameter = nx.diameter(LCC)
                    except:
                        diameter = np.nan
                    
                    betweenness = nx.betweenness_centrality(LCC)
                    closeness = nx.closeness_centrality(LCC)
                    try:
                        eigenvector = nx.eigenvector_centrality(LCC, max_iter=1000)
                    except:
                        eigenvector = {n: 0 for n in LCC.nodes()}
                    
                    avg_betweenness = np.mean(list(betweenness.values()))
                    avg_closeness = np.mean(list(closeness.values()))
                    avg_eigenvector = np.mean(list(eigenvector.values()))
                else:
                    avg_shortest_path = np.nan
                    diameter = np.nan
                    avg_betweenness = 0.0
                    avg_closeness = 0.0
                    avg_eigenvector = 0.0
            else:
                avg_shortest_path = np.nan
                diameter = np.nan
                avg_betweenness = 0.0
                avg_closeness = 0.0
                avg_eigenvector = 0.0
            
            if n_edges > 1:
                assortativity = nx.degree_assortativity_coefficient(UG)
            else:
                assortativity = np.nan
            
            movie_title = self.movie_titles.loc[self.movie_titles['MovieID'] == movie_id, 'Title'].iloc[0]
            
            records.append({
                'MovieID': movie_id,
                'Title': movie_title,
                'NumNodes': n_nodes,
                'NumEdges': n_edges,
                'AvgDegree': avg_degree,
                'Density': density,
                'AvgClustering': avg_clustering,
                'Modularity': modularity,
                'MeanEdgeSentiment': mean_sentiment,
                'StdEdgeSentiment': std_sentiment,
                'SkewEdgeSentiment': skew_sentiment,
                'KurtEdgeSentiment': kurt_sentiment,
                'AvgShortestPath': avg_shortest_path,
                'Diameter': diameter,
                'AvgBetweenness': avg_betweenness,
                'AvgCloseness': avg_closeness,
                'AvgEigenvector': avg_eigenvector,
                'Assortativity': assortativity
            })
        
        return pd.DataFrame(records)
    
    def compute_cumulative_arcs(self, movie_ids, num_points=100):
        """
        Compute average cumulative sentiment arc for a set of movie_ids.
        Normalize arcs to 'num_points' steps to allow averaging.
        Returns a numpy array of length num_points representing the average arc.
        """
        arcs = []
        for mid in movie_ids:
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == mid].sort_values('LineID')
            if len(movie_lines) == 0:
                continue
            sentiments = movie_lines['Text'].apply(lambda x: self.sia.polarity_scores(x)['compound']).values
            cumulative = np.cumsum(sentiments) / (np.arange(len(sentiments)) + 1)
            # Interpolate to fixed length
            x_old = np.linspace(0, 1, len(cumulative))
            x_new = np.linspace(0, 1, num_points)
            interp_arc = np.interp(x_new, x_old, cumulative)
            arcs.append(interp_arc)
        if len(arcs) == 0:
            return np.zeros(num_points)
        return np.mean(arcs, axis=0)
    
    def plot_average_arcs_top_bottom_20(self, top_20_ids, bottom_20_ids):
        """Plot average cumulative sentiment arcs for top 20 and bottom 20."""
        top_arc = self.compute_cumulative_arcs(top_20_ids, num_points=100)
        bottom_arc = self.compute_cumulative_arcs(bottom_20_ids, num_points=100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(top_arc, label='Top 20', color='blue', linewidth=2)
        plt.plot(bottom_arc, label='Bottom 20', color='orange', linewidth=2)
        plt.title("Average Cumulative Sentiment Arc\nTop 20 vs Bottom 20 Movies", fontsize=16, fontweight='bold')
        plt.xlabel("Normalized Dialogue Progress")
        plt.ylabel("Cumulative Sentiment")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Figures/average_arcs_top_bottom_20.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_average_arcs_per_genre(self, num_points=100):
        """Plot average cumulative sentiment arcs per individual known genre."""
        # Define the known set of genres you want to consider
        unique_genres = [
            "comedy", "romance", "adventure", "biography", "drama", "history", "action", 
            "crime", "thriller", "mystery", "sci-fi", "fantasy", "horror", "music", 
            "western", "war", "adult", "musical", "animation", "sport", "short", 
            "family", "film-noir", "documentary"
        ]



        # Initialize a dictionary: genre -> list of movie_ids
        genre_dict = {g: [] for g in unique_genres}

        # Filter movies that have genres
        movie_titles = self.movie_titles.dropna(subset=['Genres'])

        for _, row in movie_titles.iterrows():
            mid = row['MovieID']
            genres_str = row['Genres']
            if pd.isna(genres_str):
                continue
            try:
                genres_list = ast.literal_eval(genres_str)  # parse the string to a Python list
            except:
                # If parsing fails for some reason, skip this movie
                continue

            # Now genres_list is a real list like ['action', 'adventure', 'comedy']
            # Normalize the genres and add them if they match your known genres
            for g in genres_list:
                g = g.strip().lower()
                if g in genre_dict:
                    genre_dict[g].append(mid)
        
        # Temporary debugging
        # all_genres = set()  
        # for g_list in movie_titles['Genres'].dropna():
        #     for g in g_list.split('|'):
        #         all_genres.add(g.strip().lower())
        # print("Genres found in dataset:", all_genres)


        # Populate the dictionary
        for _, row in movie_titles.iterrows():
            mid = row['MovieID']
            # Split genres by '|'
            genres = str(row['Genres']).split('|')
            # Normalize and check each genre
            for g in genres:
                g = g.strip().lower()  # Ensure lowercase matching
                if g in genre_dict:    # Only add if it's one of the known genres
                    genre_dict[g].append(mid)
        
        # Plot average arcs
        plt.figure(figsize=(10, 6))
        
        # Compute and plot the average arc for each known genre
        for genre in unique_genres:
            mids = genre_dict[genre]
            if len(mids) == 0:
                # No movies in this genre, skip
                continue
            arc = self.compute_cumulative_arcs(mids, num_points=num_points)
            plt.plot(arc, label=genre)
        
        plt.title("Average Cumulative Sentiment Arc by Genre", fontsize=16, fontweight='bold')
        plt.xlabel("Normalized Dialogue Progress")
        plt.ylabel("Cumulative Sentiment")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig("Figures/average_arcs_per_genre.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Load the data
    print("Loading movie dialog data...")
    movie_lines, movie_conversations, movie_titles = load_movie_data()
    
    # Create analyzer instance
    analyzer = MovieDialogueAnalyzer(movie_lines, movie_conversations, movie_titles)
    
    # Convert Votes to numeric and drop NaNs
    movie_titles['Votes'] = pd.to_numeric(movie_titles['Votes'], errors='coerce')
    movie_titles = movie_titles.dropna(subset=['Votes'])
    
    # Top/Bottom 10 for visualization
    top_10_movies = movie_titles.nlargest(10, 'Votes')
    top_10_movie_ids = top_10_movies['MovieID'].tolist()
    
    bottom_10_movies = movie_titles.nsmallest(10, 'Votes')
    bottom_10_movie_ids = bottom_10_movies['MovieID'].tolist()

    # Top/Bottom 5
    top_5_movies = movie_titles.nlargest(5, 'Votes')
    top_5_movie_ids = top_5_movies['MovieID'].tolist()
    
    bottom_5_movies = movie_titles.nsmallest(5, 'Votes')
    bottom_5_movie_ids = bottom_5_movies['MovieID'].tolist()
    
    # Produce combined plots for top/bottom 10
    analyzer.plot_combined_sentiment_arcs(top_10_movie_ids, bottom_10_movie_ids)
    # analyzer.plot_combined_interaction_networks(top_5_movie_ids, bottom_5_movie_ids, title=False)
    # analyzer.plot_combined_emotional_keywords(top_10_movie_ids, bottom_10_movie_ids)
    
    # # Top/Bottom 100 for statistical analysis
    # top_100_movies = movie_titles.nlargest(100, 'Votes')
    # top_100_movie_ids = top_100_movies['MovieID'].tolist()
    
    # bottom_100_movies = movie_titles.nsmallest(100, 'Votes')
    # bottom_100_movie_ids = bottom_100_movies['MovieID'].tolist()
    
    # # Compute metrics for top 100 and bottom 100 movies
    # top_100_metrics_df = analyzer.compute_advanced_network_metrics(top_100_movie_ids)
    # bottom_100_metrics_df = analyzer.compute_advanced_network_metrics(bottom_100_movie_ids)

    # # Add "Group" column
    # top_100_metrics_df['Group'] = 'Top'
    # bottom_100_metrics_df['Group'] = 'Bottom'

    # # Combine into one DataFrame
    # combined_100_df = pd.concat([top_100_metrics_df, bottom_100_metrics_df], ignore_index=True)

    # # Save the DataFrame
    # combined_100_df.to_csv(r"Data\statistics\combined_100_df.csv", index=False)
    
    # print("\ncombined_100_df created and saved.")

    # # Compute top/bottom 20 sets
    # top_20_movies = movie_titles.nlargest(20, 'Votes')
    # top_20_movie_ids = top_20_movies['MovieID'].tolist()

    # bottom_20_movies = movie_titles.nsmallest(20, 'Votes')
    # bottom_20_movie_ids = bottom_20_movies['MovieID'].tolist()

    # # Plot average arcs for top/bottom 20
    # analyzer.plot_average_arcs_top_bottom_20(top_20_movie_ids, bottom_20_movie_ids)

    # # Plot average arcs per genre (using all movies available)
    # analyzer.plot_average_arcs_per_genre(num_points=100)
    
    # print("Added average cumulative arcs for top/bottom 20 and per genre.")
    # print("Check the 'Figures' directory for the new figures.")
