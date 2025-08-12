import networkx as nx
import matplotlib.pyplot as plt
import itertools
from typing import List, Tuple, Dict, Set
import random
import pandas as pd
import topohub
import subprocess

from sklearn.cluster import KMeans
import numpy as np

def initial_partition_knn(G: nx.Graph, k: int) -> Dict[int, int]:
    nodes = list(G.nodes())
    
    # Example feature: degree and clustering coefficient
    degrees = np.array([G.degree(n) for n in nodes]).reshape(-1, 1)
    clustering = np.array([nx.clustering(G, n) for n in nodes]).reshape(-1, 1)

    # Combine features into a single matrix
    features = np.hstack((degrees, clustering))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    # Map node to partition
    node_to_part = {node: int(label) for node, label in zip(nodes, labels)}
    return node_to_part

def initial_partition_worse(G: nx.Graph, k: int) -> Dict[int, int]:
    node_to_part = {}
    for node in G.nodes():
        node_to_part[node] = random.randint(0, k - 1)
    return node_to_part

def initial_partition(G: nx.Graph, k: int) -> Dict[int, int]:
    node_to_part = {}
    unassigned = set(G.nodes())
    seeds = random.sample(sorted(unassigned), k)
    
    for i, seed in enumerate(seeds):
        node_to_part[seed] = i
        unassigned.remove(seed)

    for node in unassigned:
        neighbors = list(G.neighbors(node))
        neighbor_parts = [node_to_part.get(n) for n in neighbors if n in node_to_part]
        
        if neighbor_parts:
            node_to_part[node] = max(set(neighbor_parts), key=neighbor_parts.count)
        else:
            node_to_part[node] = random.randint(0, k - 1)

    return node_to_part

def compute_cut_edges(G: nx.Graph, partition: Dict[int, int]) -> Set[Tuple[int, int]]:
    return {(u, v) for u, v in G.edges() if partition[u] != partition[v]}

def is_valid_move(G, partition, node, target_part, max_vertices, max_edges) -> bool:
    current_partition = {n for n, p in partition.items() if p == target_part}
    if len(current_partition) + 1 > max_vertices:
        return False
    induced = G.subgraph(current_partition | {node})
    return nx.is_connected(induced) and induced.number_of_edges() <= max_edges

def kernighan_lin_with_constraints(G: nx.Graph, partition: Dict[int, int], max_vertices: int, max_edges: int, max_iter=100) -> Dict[int, int]:
    for _ in range(max_iter):
        improvement = False
        for u in G.nodes():
            current_part = partition[u]
            best_gain = 0
            best_target = current_part
            for target_part in set(partition.values()):
                if target_part == current_part:
                    continue
                if not is_valid_move(G, partition, u, target_part, max_vertices, max_edges):
                    continue
                partition[u] = target_part
                cut_edges = len(compute_cut_edges(G, partition))
                partition[u] = current_part
                gain = len(compute_cut_edges(G, partition)) - cut_edges
                if gain > best_gain:
                    best_gain = gain
                    best_target = target_part
            if best_target != current_part:
                partition[u] = best_target
                improvement = True
        if not improvement:
            break
    return partition

def build_subgraphs_from_partition(G: nx.Graph, partition: Dict[int, int]) -> List[nx.Graph]:
    parts = {}
    for node, p in partition.items():
        parts.setdefault(p, set()).add(node)
    return [G.subgraph(nodes).copy() for nodes in parts.values()]

def draw_partitioned_graph(G: nx.Graph, partition: Dict[int, int]):
    pos = nx.spring_layout(G, seed=42)
    colors = plt.cm.tab10.colors
    unique_parts = sorted(set(partition.values()))

    for i in unique_parts:
        nodes = [n for n, p in partition.items() if p == i]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors[i % len(colors)]] * len(nodes), label=f"Emulation Plat. {i}")

    intra_edges = [(u, v) for u, v in G.edges() if partition[u] == partition[v]]
    cut_edges = [(u, v) for u, v in G.edges() if partition[u] != partition[v]]

    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', style='dashed', width=2)
    nx.draw_networkx_labels(G, pos)
    plt.title("GhostTwin Partitioning (Cut edges in red)")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":

    #G = nx.barabasi_albert_graph(n=20, m=3)
    #repo_url = "https://github.com/piotrjurkiewicz/topohub.git"
    #clone_dir = "topohub"
    # Clone the repo
    #subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    # Execute 'ls' and capture output
    result = subprocess.run(["ls", "topohub/data/topozoo/"], capture_output=True, text=True, check=True)

    # Split output into a list of filenames
    file_list = result.stdout.strip().split("\n")
    
    json_files = [f.removesuffix(".json") for f in file_list if f.endswith(".json")]

    print("Number of topo: " + str(len(json_files)))
    
    
    results = []
    #k = 5 

    for file in json_files:

        for k in range(1, 11):

            print(file)
            topo = topohub.get('topozoo/' + file)
            
            best = None
            bestCutEdge = 99999
            
            for i in range(1, 10):
                
                G = nx.node_link_graph(topo)
                if G.number_of_nodes() < k:
                    continue    
                max_vertices = 10
                max_edges = 10

                initial = initial_partition(G, k)
                #initial = initial_partition_worse(G, k)
                #initial = initial_partition_knn(G, k)
                subgraphs = build_subgraphs_from_partition(G, initial)
                

                refined = kernighan_lin_with_constraints(G, initial, max_vertices, max_edges)
                #draw_partitioned_graph(G, refined)
                #refined = initial 

                subgraphs = build_subgraphs_from_partition(G, refined)
                cut_edges = [(u, v) for u, v in G.edges() if refined[u] != refined[v]]
                
                if len(cut_edges) < bestCutEdge:
                    #verifiy if all partitions are valid
                    partition_ok = True
                    for i, part in enumerate(subgraphs):
                        if len(part.nodes()) > max_vertices:
                            partition_ok = False
                            break

                    if partition_ok : 
                        bestCutEdge = len(cut_edges)
                        best = refined

            
            if best is None:
                print(f"No valid partition found for {file}")
                continue

            refined = best
            subgraphs = build_subgraphs_from_partition(G, refined)

            
            partition_data = []
            for i, part in enumerate(subgraphs):
                partition_data.append({
                    "Graph": file,
                    "CutEdges": len(cut_edges),
                    "k": k,
                    "Partition": i,
                    "Vertices": sorted(part.nodes()),
                    "Edges": sorted(part.edges()),
                    "Size (V)": part.number_of_nodes(),
                    "NumPartitions": len(subgraphs),
                    "Size (E)": part.number_of_edges()
                })

            df = pd.DataFrame(partition_data)
            results.append(df)       
            print(df.to_string(index=False))

    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv("results-10-10-lk.csv", index=False)



