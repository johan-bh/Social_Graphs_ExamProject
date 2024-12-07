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
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
import string
from adjustText import adjust_text

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
        """Create a directed interaction network for a specific movie."""
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
        
        return MG, char_names

    def plot_all_sentiment_arcs(self, movie_ids):
        """Plot cumulative sentiment arcs with bold, large titles and capitalized movie titles."""
        n_movies = len(movie_ids)
        n_cols = 5
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
        axes = axes.flatten()
        
        for ax, movie_id in zip(axes, movie_ids):
            movie_lines = self.movie_lines[self.movie_lines['MovieID'] == movie_id].sort_values('LineID')
            movie_lines['Sentiment'] = movie_lines['Text'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
            movie_lines['CumulativeAverage'] = movie_lines['Sentiment'].expanding().mean()
            
            ax.plot(range(len(movie_lines)), movie_lines['CumulativeAverage'], color='red', linestyle='--', linewidth=2)
            ax.fill_between(range(len(movie_lines)), movie_lines['CumulativeAverage'], color='red', alpha=0.1)
            
            title = self.movie_titles[self.movie_titles['MovieID'] == movie_id]['Title'].iloc[0]
            ax.set_title(title.title(), fontsize=14, fontweight='bold')
            ax.set_xlabel('Dialogue Sequence')
            ax.set_ylabel('Cumulative Sentiment')
            ax.grid(True, alpha=0.3)
        
        for ax in axes[len(movie_ids):]:
            ax.remove()
        
        plt.suptitle('Cumulative Sentiment Arcs Across Movies', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('Figures/sentiment_arcs_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_all_interaction_networks(self, movie_ids):
        """Plot interaction networks and only label top nodes by degree to reduce crowding."""
        n_movies = len(movie_ids)
        n_cols = 5
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        fig.suptitle("Character Interaction Networks for Top Movies\nEdge colors indicate sentiment (blue=negative, red=positive)",
                     fontsize=22, fontweight='bold', y=0.95)
        
        axes = axes.flatten()
        
        for idx, (ax, movie_id) in enumerate(zip(axes, movie_ids)):
            movie = self.movie_titles[self.movie_titles['MovieID'] == movie_id].iloc[0]
            
            MG, char_names = self.create_interaction_network(movie_id)
            
            pos = nx.kamada_kawai_layout(MG)
            
            node_size = 300
            edge_colors = [MG[u][v]['sentiment'] for u, v in MG.edges()]
            
            # Draw nodes first
            nx.draw_networkx_nodes(
                MG, pos,
                node_color='lightgray',
                node_size=node_size,
                alpha=0.8,
                ax=ax
            )
            
            # Draw edges with thicker width
            nx.draw_networkx_edges(
                MG, pos,
                edge_color=edge_colors,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=-1,
                edge_vmax=1,
                width=2,
                alpha=1.0,
                arrows=False,
                ax=ax
            )
            
            # Label only top N nodes by degree to reduce clutter
            degrees = dict(MG.degree())
            # Sort nodes by degree and pick top 8
            top_nodes = [n for n, deg in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:8]]
            labels = {node: char_names.get(node, str(node)) for node in top_nodes}
            
            texts = []
            for node in top_nodes:
                x, y = pos[node]
                text = ax.text(x, y, labels[node],
                               fontsize=8,
                               fontweight='bold',
                               horizontalalignment='center',
                               verticalalignment='center',
                               bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
                texts.append(text)
            
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
            
            ax.set_title(f"{movie['Title'].title()}\n{MG.number_of_nodes()} chars, {MG.number_of_edges()} interactions",
                         fontsize=10, fontweight='bold')
            ax.axis('off')
        
        for ax in axes[len(movie_ids):]:
            ax.remove()
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Edge Sentiment Score', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.93])
        plt.savefig('Figures/interaction_networks_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_all_emotional_keywords(self, movie_ids):
        """Plot emotional keywords with improved title style."""
        n_movies = len(movie_ids)
        n_cols = 5
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        axes = axes.flatten()
        
        for ax, movie_id in zip(axes, movie_ids):
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
                
                words_plot = [word for word, data in sorted_words]
                sentiments = [data['sentiment'] for word, data in sorted_words]
                colors = ['red' if s < 0 else 'green' for s in sentiments]
                ax.barh(words_plot, sentiments, color=colors, alpha=0.6)
                ax.set_xlim(-1, 1)
            else:
                ax.text(0.5, 0.5, 'No Significant Keywords Found', ha='center', va='center')
            
            title = self.movie_titles[self.movie_titles['MovieID'] == movie_id]['Title'].iloc[0]
            ax.set_title(title.title(), fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Keywords')
        
        for ax in axes[len(movie_ids):]:
            ax.remove()
        
        plt.suptitle('Emotional Keywords Across Movies', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('Figures/emotional_keywords_combined.png', dpi=300, bbox_inches='tight')
        plt.close()

# Usage example:
if __name__ == "__main__":
    # Load the data
    print("Loading movie dialog data...")
    movie_lines, movie_conversations, movie_titles = load_movie_data()
    
    # Create analyzer instance
    analyzer = MovieDialogueAnalyzer(movie_lines, movie_conversations, movie_titles)
    
    # Get top 10 movies
    print("\nAnalyzing top movies by vote count...")
    movie_titles['Votes'] = pd.to_numeric(movie_titles['Votes'], errors='coerce')
    top_movies = movie_titles.dropna(subset=['Votes']).nlargest(10, 'Votes')
    movie_ids = top_movies['MovieID'].tolist()
    
    # Generate combined analyses for all movies
    print("\nGenerating combined analyses for all movies...")
    
    # Plot sentiment arcs for all movies
    analyzer.plot_all_sentiment_arcs(movie_ids)
    
    # Plot interaction networks for all movies with reduced clutter
    analyzer.plot_all_interaction_networks(movie_ids)
    
    # Plot emotional keywords for all movies
    analyzer.plot_all_emotional_keywords(movie_ids)
    
    print("\nAnalysis complete! Check the 'Figures' directory for visualizations.")
