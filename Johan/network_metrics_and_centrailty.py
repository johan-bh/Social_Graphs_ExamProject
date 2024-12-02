import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a Seaborn theme for aesthetics
sns.set(style="whitegrid")

# Load the bipartite graph dataset
file_path = r"C:\Users\jbh\Desktop\Social_Graphs_ExamProject\Data\movie_casts_sample.csv"
movie_casts = pd.read_csv(file_path)

# Create a bipartite graph
B = nx.Graph()

# Add movie and actor nodes
movies = movie_casts['movie_title'].unique()
actors = movie_casts['actor_name'].unique()

B.add_nodes_from(movies, bipartite=0)  # Movie nodes
B.add_nodes_from(actors, bipartite=1)  # Actor nodes

# Add edges between movies and actors
for _, row in movie_casts.iterrows():
    B.add_edge(row['movie_title'], row['actor_name'])

# Projected in-degrees and out-degrees
actor_degrees = [B.degree(n) for n in actors]
movie_degrees = [B.degree(n) for n in movies]

# Generate Barabási–Albert and Random graphs for comparison
n = len(movies) + len(actors)
m = B.number_of_edges() // len(movies)  # Approximate edges per node for BA
ba_graph = nx.barabasi_albert_graph(n, m)
random_graph = nx.gnm_random_graph(n, B.number_of_edges())

# Convert BA graph to directed
directed_ba_graph = nx.DiGraph()
for u, v in ba_graph.edges():
    if np.random.rand() > 0.5:
        directed_ba_graph.add_edge(u, v)  # u -> v
    else:
        directed_ba_graph.add_edge(v, u)  # v -> u

# Compute in-degree and out-degree distributions for BA and Random graphs
ba_in_degrees = [directed_ba_graph.in_degree(n) for n in directed_ba_graph.nodes()]
ba_out_degrees = [directed_ba_graph.out_degree(n) for n in directed_ba_graph.nodes()]
random_in_degrees = [random_graph.degree(n) for n in random_graph.nodes()]


original_clustering = nx.average_clustering(B)
random_clustering = nx.average_clustering(random_graph)
ba_clustering = nx.average_clustering(ba_graph)
print(f"Original: {original_clustering}, Random: {random_clustering}, BA: {ba_clustering}")


from community import community_louvain
partition = community_louvain.best_partition(B)
modularity = community_louvain.modularity(partition, B)
print(f"Modularity: {modularity}")


import powerlaw
degree_data = [d for n, d in B.degree()]
fit = powerlaw.Fit(degree_data)
print(f"Power law alpha: {fit.alpha}, Lognormal sigma: {fit.sigma}")


# Get connected components
components = [B.subgraph(c).copy() for c in nx.connected_components(B)]

# Compute the average shortest path length for each component
component_path_lengths = [
    nx.average_shortest_path_length(component)
    for component in components
]

# Compute the weighted average based on the size of each component
total_nodes = sum(len(c) for c in components)
weighted_avg_path_length = sum(
    len(c) * nx.average_shortest_path_length(c) for c in components
) / total_nodes

print(f"Weighted Average Path Length: {weighted_avg_path_length}")

# Extract the largest connected component
largest_cc = max(nx.connected_components(B), key=len)
largest_cc_subgraph = B.subgraph(largest_cc).copy()

# Compute the average shortest path length for the largest component
avg_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
print(f"Average Path Length (Largest Connected Component): {avg_path_length}")


def plot_degree_distributions(real_data, ba_data, random_data, title, xlabel):
    num_bins = 50

    # Compute histograms with consistent bins for fair comparison
    bins = np.linspace(0, max(max(real_data), max(ba_data), max(random_data)), num_bins)
    real_hist, _ = np.histogram(real_data, bins=bins)
    ba_hist, _ = np.histogram(ba_data, bins=bins)
    random_hist, _ = np.histogram(random_data, bins=bins)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Adjust bar width
    bar_width = (bins[1] - bins[0]) * 0.8  # Consistent width for all bars

    # Plot histograms with offsets to reduce overlap
    ax.bar(bins[:-1], real_hist, width=bar_width, color="blue", alpha=0.5, label="Bipartite Network", align='edge')
    ax.bar(bins[:-1] - bar_width, ba_hist, width=bar_width, color="red", alpha=0.5, label="Barabási-Albert Graph", align='edge')
    ax.bar(bins[:-1] + bar_width, random_hist, width=bar_width, color="green", alpha=0.5, label="Random Graph", align='edge')

    # Add labels, title, and legend
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_yscale("log")  # Log scale for better visualization
    ax.legend(fontsize=12)

    # Add grid for better readability
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Ensure tight layout and save the plot
    plt.tight_layout()
    plt.savefig(f'Figures/{title.replace(" ", "_")}_degree_distribution.png', dpi=300)
    plt.show()


# Plot in-degree and out-degree distributions
plot_degree_distributions(actor_degrees, ba_in_degrees, random_in_degrees, "In-degree Distribution Comparison", "In-degree")
plot_degree_distributions(movie_degrees, ba_out_degrees, random_in_degrees, "Out-degree Distribution Comparison", "Out-degree")