import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, util
from skimage.morphology import skeletonize, thin
import networkx as nx
from skimage.draw import line
from collections import Counter
from collections import defaultdict
from scipy.spatial import distance_matrix

def build_graph(skeleton, connectivity=2):
    """
    Convert a skeletonized image to a NetworkX graph.
    
    Parameters:
    - skeleton: 2D binary numpy array.
    - connectivity: int, 1 for 4-connectivity, 2 for 8-connectivity.
    
    Returns:
    - G: NetworkX graph.
    """
    G = nx.Graph()
    rows, cols = skeleton.shape
    # Define neighbor offsets based on connectivity
    if connectivity == 1:
        neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
    elif connectivity == 2:
        neighbors = [(-1,0), (1,0), (0,-1), (0,1),
                     (-1,-1), (-1,1), (1,-1), (1,1)]
    else:
        raise ValueError("Connectivity must be 1 or 2")
    
    # Add nodes and edges
    for y in range(rows):
        for x in range(cols):
            if skeleton[y, x]:
                G.add_node((y, x))
                for dy, dx in neighbors:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx_ < cols:
                        if skeleton[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))
    return G

def merge_close_subgraphs(G, dist_threshold=2.0):
    """
    Merge subgraphs in a NetworkX graph if they are within 'dist_threshold'.
    Effectively 'bridges' or unifies edges that are close together.
    
    Parameters:
    - G: NetworkX graph (undirected).
    - dist_threshold: float, distance below which two subgraphs get merged.
    
    Returns:
    - merged_graph: A new NetworkX graph with edges added
                    to connect components that are close.
    """
    # Make a copy so we don't mutate the original graph
    merged_graph = G.copy()
    
    # Step 1: Find connected components
    components = list(nx.connected_components(G))
    
    # Step 2: For each pair of distinct components, find the minimum distance
    # between any node in one component and any node in the other.
    # If min_dist < dist_threshold, add an edge between the closest nodes.
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp_i = components[i]
            comp_j = components[j]
            min_dist = float('inf')
            closest_pair = (None, None)
            
            for node_i in comp_i:
                for node_j in comp_j:
                    dist = np.linalg.norm(np.array(node_i) - np.array(node_j))
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node_i, node_j)
            
            if min_dist < dist_threshold:
                # Merge by adding an edge between the closest nodes.
                # This effectively unifies these two subgraphs.
                merged_graph.add_edge(closest_pair[0], closest_pair[1])
    
    return merged_graph

def get_edge_pixel_coords(labeled_image, label):
    """
    Returns a list of pixel coordinates belonging to 'label' in 'labeled_image'.
    """
    coords = np.argwhere(labeled_image == label)
    # coords is shape (N,2), each row [r,c]; convert to list of tuples if needed:
    # coords_list = [tuple(row) for row in coords]
    return coords

def measure_edge_distance(coords1, coords2):
    """
    Compute a basic measure of distance between two sets of edge coordinates.
    For simplicity, we'll use the average of minimal distances (a rough measure).
    """
    if coords1.size == 0 or coords2.size == 0:
        return np.inf
    dist_mat = distance_matrix(coords1, coords2)
    # E.g. take the average of the minimum distances in both directions
    d1 = dist_mat.min(axis=1).mean()  # average distance from coords1 to coords2
    d2 = dist_mat.min(axis=0).mean()  # average distance from coords2 to coords1
    return (d1 + d2) / 2.0

def average_edge(coords1, coords2):
    """
    Construct a naive 'average' path between coords1 and coords2.
    This is a very rough approach:
      - Combine both sets of points
      - Take the convex hull or bounding box
      - Possibly do a thinning or shortest path from start to end
    Here we just combine them, then we'll pick out a skeleton via connectivity.

    Returns: np.array of shape (N, 2) of the final path coordinates.
    """
    combined = np.vstack([coords1, coords2])
    # One naive approach: just take the unique coordinates and call that our set:
    combined_unique = np.unique(combined, axis=0)
    return combined_unique

def find_start_end_for_label(labeled_image, label, junctions):
    """
    Returns the (start_junction, end_junction) for the given 'label' by checking
    which junction(s) it touches. If it doesn't have exactly two distinct endpoints,
    returns None, None.
    
    'junctions' is typically a set or list of (row, col) coordinates
    that you recognized as junction points in your graph.
    """
    coords = get_edge_pixel_coords(labeled_image, label)
    coords_set = {tuple(c) for c in coords}

    # Which junction points are in this edge?
    junction_hits = coords_set.intersection(junctions)
    # For a normal "simple" edge (no cycles), we expect 2 endpoints if it
    # connects two distinct junctions. If more or fewer, might be a cycle or
    # a partial edge.
    if len(junction_hits) == 2:
        return tuple(sorted(junction_hits))
    else:
        return None, None

def merge_duplicate_edges_with_average(
    labeled_image,
    junctions,
    dist_threshold=3.0,
    background_label=0
):
    """
    Merges edges that share the same start and end junction
    and are 'close' (less than dist_threshold apart on average),
    by replacing them with a single 'average' edge.

    labeled_image: int 2D array from trace_edges (each edge has a unique >0 label).
    junctions: set or list of (r,c) junction coordinates.
    dist_threshold: float, threshold for deciding if two edges are close.
    background_label: int, usually 0 for background.

    Returns:
    - merged_labeled: updated 2D array with merges performed.
    """
    merged_labeled = labeled_image.copy()
    labels = np.unique(merged_labeled)
    labels = labels[labels != background_label]  # skip background

    # 1) For each label, find (start_junction, end_junction) + store coords
    edge_info = {}
    for lbl in labels:
        sj, ej = find_start_end_for_label(merged_labeled, lbl, junctions)
        if sj is not None and ej is not None:
            coords = get_edge_pixel_coords(merged_labeled, lbl)
            edge_info[lbl] = {
                'start': sj,
                'end': ej,
                'coords': coords
            }
        else:
            # Could be a cycle or something else, skip or store separately
            pass

    # 2) Group edges by their (start, end) pairs
    edges_by_pair = defaultdict(list)
    for lbl, info in edge_info.items():
        pair = (info['start'], info['end'])
        # To avoid direction issues, we ensure the start<end is consistent:
        # (It should already be sorted in find_start_end_for_label)
        edges_by_pair[pair].append(lbl)

    # We'll need to track which labels got merged & replaced
    used_labels = set()  # keep track of edges we've already merged

    # We'll generate new labels for merges
    new_label_counter = merged_labeled.max() + 1

    # 3) For each (start, end) pair, see if multiple edges exist
    for pair, lbl_list in edges_by_pair.items():
        if len(lbl_list) < 2:
            continue  # nothing to merge if there's only 1 edge

        # Potentially, we could do all-pairs merges. For simplicity,
        # we show a simple approach that merges the entire group
        # into one edge if *all* are close to each other, or merges
        # them pairwise if they pass the threshold. 
        # We'll do pairwise checks here:

        merged_edges = []
        while lbl_list:
            current_lbl = lbl_list.pop()
            if current_lbl in used_labels:
                continue

            current_coords = edge_info[current_lbl]['coords']
            to_merge = [current_lbl]
            # We collect edges that are "close" to current_lbl
            still_unassigned = []
            for other_lbl in lbl_list:
                if other_lbl in used_labels:
                    continue
                other_coords = edge_info[other_lbl]['coords']
                dist = measure_edge_distance(current_coords, other_coords)
                if dist < dist_threshold:
                    to_merge.append(other_lbl)
                else:
                    still_unassigned.append(other_lbl)
            # Update lbl_list to those not merged with current_lbl
            lbl_list = still_unassigned

            if len(to_merge) > 1:
                # Merge all in to_merge
                all_coords = []
                for lbl_m in to_merge:
                    used_labels.add(lbl_m)
                    all_coords.append(edge_info[lbl_m]['coords'])
                # Flatten
                all_coords = np.vstack(all_coords)

                # Construct an "average" path
                # For simplicity, let's just do "unique + minimal skeleton"
                combined_unique = np.unique(all_coords, axis=0)

                # Remove old labels from the image
                for lbl_m in to_merge:
                    merged_labeled[merged_labeled == lbl_m] = background_label

                # Now we create a new label for the merged edge
                new_label = new_label_counter
                new_label_counter += 1

                # Mark these coordinates in the image
                for (r, c) in combined_unique:
                    merged_labeled[r, c] = new_label

                merged_edges.append(new_label)
            else:
                # Only one label in to_merge => no merge needed
                # push it back if we want to keep it
                pass

    return merged_labeled

def identify_junctions_and_endpoints(G):
    """
    Identify junctions and endpoints in the graph.
    
    Parameters:
    - G: NetworkX graph.
    
    Returns:
    - junctions: list of node tuples with degree >=3.
    - endpoints: list of node tuples with degree ==1.
    """
    junctions = [node for node, degree in G.degree() if degree >=3]
    endpoints = [node for node, degree in G.degree() if degree ==1]
    return junctions, endpoints

def trace_edges(G, junctions, endpoints, height, width):
    """
    Trace each edge from endpoints or junctions to other endpoints or junctions.
    """
    labeled_image = np.zeros(
        (height, width),
        dtype=np.int32
    )
    
    label = 1
    visited = set()
    
    # Combine endpoints and junctions into a single list of "start" nodes.
    start_nodes = list(set(endpoints + junctions))
    
    for start_node in start_nodes:
        if start_node in visited:
            continue
        
        stack = [(start_node, [start_node])]
        
        while stack:
            current, path = stack.pop()
            visited.add(current)
            neighbors = list(G.neighbors(current))
            
            for neighbor in neighbors:
                if (neighbor in visited) and (neighbor != path[0]):
                    continue
                if (neighbor in junctions) or (neighbor in endpoints):
                    # We reached another endpoint or junction.
                    path_extended = path + [neighbor]
                    # Label the entire path.
                    rr, cc = zip(*path_extended)
                    labeled_image[rr, cc] = label
                    label += 1
                else:
                    stack.append((neighbor, path + [neighbor]))
    
    # Handle cycles
    cycles = [c for c in nx.cycle_basis(G) if len(c) > 0]
    for cycle in cycles:
        # Check if none of these cycle nodes were already labeled
        if any(node in visited for node in cycle):
            # If you prefer, skip if already visited
            # or do a partial labeling check
            continue
        # num_visited = 0
        # for node in cycle:
        #     if node in visited:
        #         num_visited += 1
        # if num_visited >= 4: # if there is only 1 visited node, we will label it, otherwise we skip
        #     continue
        rr, cc = zip(*cycle)
        labeled_image[rr, cc] = label
        label += 1
    
    return labeled_image

def split_connected_components(edge_map, connectivity=2):
    """
    Split connected components at junctions and assign unique labels to each edge.
    
    Parameters:
    - edge_map: 2D binary numpy array.
    - connectivity: int, 1 for 4-connectivity, 2 for 8-connectivity.
    
    Returns:
    - labeled_edges: 2D numpy array with unique labels for each edge.
    """
    # Step 1: Skeletonize the edge map
    # skeleton = skeletonize(edge_map).astype(np.uint8)
    skeleton = thin(edge_map).astype(np.uint8)
    
    # Step 2: Build graph from skeleton
    G = build_graph(skeleton, connectivity=connectivity)

    # G = merge_close_subgraphs(G, dist_threshold=5.0)
    
    # Step 3: Identify junctions and endpoints
    junctions, endpoints = identify_junctions_and_endpoints(G)
    
    # Step 4: Trace and label edges
    height, width = edge_map.shape[:2]
    labeled_edges = trace_edges(G, junctions, endpoints, height, width)

    # merged_labeled = merge_duplicate_edges_with_average(
    #     labeled_image=labeled_edges,
    #     junctions=junctions,
    #     dist_threshold=8.0
    # )
    
    # return merged_labeled, junctions, endpoints
    return labeled_edges, junctions, endpoints

def extract_and_count_values(matrix, binary_mask):
    """
    Extracts values from the input matrix based on the binary mask and
    returns the statistics (counts) of the unique values.
    
    Parameters:
        matrix (numpy.ndarray): A 2D or 3D numpy array containing the values.
        binary_mask (numpy.ndarray): A binary mask of the same shape as the matrix.
                                     Non-zero entries in the mask indicate points of interest.
    
    Returns:
        dict: A dictionary where keys are the unique values in the masked region and 
              values are the counts of those unique values.
    """
    # Ensure the mask and matrix have the same shape
    if matrix.shape != binary_mask.shape:
        raise ValueError("The matrix and binary mask must have the same shape.")
    
    # Extract the values based on the mask
    masked_values = matrix[binary_mask > 0]
    
    # Count the occurrences of each unique value
    value_counts = dict(Counter(masked_values))
    count_values = {v: k for k, v in value_counts.items()}
    
    # Convert to a dictionary and return
    return value_counts, count_values

def decide_value_and_confidence(count_values):
    # confidence: 1 low conf, 2 medium conf, 3 high conf
    total_counts = np.sum(list(count_values.keys()))
    idx_ascending = np.argsort(list(count_values.keys()))
    max_idx = idx_ascending[-1]
    count = list(count_values.keys())[max_idx]
    value = count_values[count]

    if len(count_values) == 1:
        return value, 3
    elif count > 0.6 * total_counts:
        return value, 3
    else:
        idx_2 = idx_ascending[-2]
        count2 = list(count_values.keys())[idx_2]
        value2 = count_values[count]

        if count + count2 > 0.9 * total_counts:
            return max(value, value2), 2
        else:
            return value, 1

if __name__ == "__main__":
    # Create a synthetic edge map with multiple overlapping lines and junctions
    rows, cols = 256, 256
    edge_map = np.zeros((rows, cols), dtype=np.uint8)
    
    # Draw multiple lines intersecting at a central junction
    lines = [
        ((50, 50), (200, 200)),    # Diagonal 1
        ((50, 200), (200, 50)),    # Diagonal 2
        ((128, 50), (128, 200)),   # Vertical
        ((50, 128), (200, 128)),   # Horizontal
        ((70, 70), (180, 180)),    # Diagonal 3
        ((70, 180), (180, 70))     # Diagonal 4
    ]
    
    for start, end in lines:
        rr, cc = line(start[0], start[1], end[0], end[1])
        edge_map[rr, cc] = 1
    
    # Introduce some noise
    noise = np.random.rand(rows, cols) < 0.001
    edge_map = np.logical_or(edge_map, noise).astype(np.uint8)
    
    # Display the original edge map
    plt.figure(figsize=(6,6))
    plt.imshow(edge_map, cmap='gray')
    plt.title('Original Edge Map with Junctions')
    plt.axis('off')
    plt.show()
    
    # Apply the splitting function
    labeled_edges = split_connected_components(edge_map, connectivity=2)
    
    # Display the labeled edge components
    plt.figure(figsize=(6,6))
    plt.imshow(labeled_edges, cmap='nipy_spectral')
    plt.title('Labeled Edge Components')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    # Print number of unique components
    num_labels = labeled_edges.max()
    print(f"Number of distinct edge components: {num_labels}")
    
    # Optionally, visualize each component separately
    plt.figure(figsize=(12, 12))
    for label in range(1, num_labels + 1):
        component = (labeled_edges == label).astype(np.uint8)
        plt.subplot(5, 5, label)
        plt.imshow(component, cmap='gray')
        plt.title(f'Component {label}')
        plt.axis('off')
        if label >= 25:
            break  # Limit to first 25 components for visualization
    plt.tight_layout()
    plt.show()
