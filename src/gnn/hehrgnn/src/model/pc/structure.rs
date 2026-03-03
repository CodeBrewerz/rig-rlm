//! Structure learning for Probabilistic Circuits.
//!
//! Builds a PC structure from data using:
//! - **HCLT**: Hidden Chow-Liu Tree (mutual information → MST → hidden variable tree)
//! - **Naive**: Fully-factorized baseline
//! - **from_variables**: User-specified variable structure

use super::distribution::Distribution;
use super::node::CircuitBuilder;

/// Build a naive fully-factorized PC (product of independent categoricals).
///
/// P(X1, ..., Xn) = P(X1) × P(X2) × ... × P(Xn)
pub fn naive(num_vars: usize, num_categories: usize) -> CircuitBuilder {
    let mut b = CircuitBuilder::new();
    let inputs: Vec<_> = (0..num_vars)
        .map(|v| b.add_input(v, Distribution::uniform_categorical(num_categories)))
        .collect();
    if inputs.len() > 1 {
        b.add_product(inputs);
    }
    b
}

/// Build an HCLT-like PC from data.
///
/// 1. Estimate pairwise mutual information from data
/// 2. Build a maximum spanning tree (Chow-Liu tree)
/// 3. For each tree edge (parent, child), create a hidden "sum" mixture
///    connecting the two variables' input distributions.
///
/// `data`: [num_samples][num_vars] — integer-valued observations.
/// `num_categories`: number of categories per variable.
/// `num_latents`: number of latent components (mixture size at each tree node).
pub fn hclt(data: &[Vec<usize>], num_categories: usize, num_latents: usize) -> CircuitBuilder {
    let num_vars = data[0].len();
    let num_samples = data.len();

    if num_vars <= 1 {
        return naive(num_vars, num_categories);
    }

    // Step 1: Compute pairwise mutual information
    let mi = mutual_information(data, num_categories);

    // Step 2: Maximum spanning tree (Kruskal's / Prim's)
    let tree = maximum_spanning_tree(&mi, num_vars);

    // Step 3: Root the tree at node 0 and build PC
    let (children_map, root) = root_tree(&tree, num_vars);

    // Step 4: Build PC recursively
    let mut b = CircuitBuilder::new();
    build_hclt_recursive(
        &mut b,
        root,
        &children_map,
        num_categories,
        num_latents,
        data,
    );

    b
}

/// Compute pairwise mutual information matrix.
fn mutual_information(data: &[Vec<usize>], num_cats: usize) -> Vec<Vec<f64>> {
    let n_vars = data[0].len();
    let n = data.len() as f64;
    let mut mi = vec![vec![0.0; n_vars]; n_vars];

    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            // Count joint and marginal frequencies
            let mut joint = vec![vec![0u64; num_cats]; num_cats];
            let mut count_i = vec![0u64; num_cats];
            let mut count_j = vec![0u64; num_cats];

            for sample in data {
                let vi = sample[i].min(num_cats - 1);
                let vj = sample[j].min(num_cats - 1);
                joint[vi][vj] += 1;
                count_i[vi] += 1;
                count_j[vj] += 1;
            }

            // MI = Σ p(x,y) log [p(x,y) / (p(x)p(y))]
            let mut mi_val = 0.0;
            for a in 0..num_cats {
                for b_val in 0..num_cats {
                    let pxy = joint[a][b_val] as f64 / n;
                    let px = count_i[a] as f64 / n;
                    let py = count_j[b_val] as f64 / n;
                    if pxy > 1e-12 && px > 1e-12 && py > 1e-12 {
                        mi_val += pxy * (pxy / (px * py)).ln();
                    }
                }
            }
            mi[i][j] = mi_val;
            mi[j][i] = mi_val;
        }
    }
    mi
}

/// Build maximum spanning tree using Kruskal's algorithm.
fn maximum_spanning_tree(mi: &[Vec<f64>], n: usize) -> Vec<(usize, usize, f64)> {
    // Sort all edges by weight (descending for max spanning tree)
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j, mi[i][j]));
        }
    }
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    let mut tree = Vec::new();
    for (u, v, w) in edges {
        let pu = find(&mut parent, u);
        let pv = find(&mut parent, v);
        if pu != pv {
            parent[pu] = pv;
            tree.push((u, v, w));
            if tree.len() == n - 1 {
                break;
            }
        }
    }
    tree
}

/// Root the tree at a given node, returning adjacency as children map.
fn root_tree(tree: &[(usize, usize, f64)], n: usize) -> (Vec<Vec<usize>>, usize) {
    let root = 0;
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(u, v, _) in tree {
        adj[u].push(v);
        adj[v].push(u);
    }

    // BFS to root the tree
    let mut children: Vec<Vec<usize>> = vec![vec![]; n];
    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    visited[root] = true;

    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                children[u].push(v);
                queue.push_back(v);
            }
        }
    }

    (children, root)
}

/// Recursively build HCLT circuit.
///
/// At each tree node: create `num_latents` components, each a product of
/// the node's input distribution and children's sum nodes.
fn build_hclt_recursive(
    b: &mut CircuitBuilder,
    var: usize,
    children_map: &[Vec<usize>],
    num_cats: usize,
    num_latents: usize,
    data: &[Vec<usize>],
) -> usize {
    if children_map[var].is_empty() {
        // Leaf: single input node with empirical distribution
        let dist = empirical_distribution(data, var, num_cats);
        return b.add_input(var, dist);
    }

    // Create num_latents components
    let mut components = Vec::new();
    for _k in 0..num_latents {
        // This component: product of (this var's input) × (child subtrees)
        let input = b.add_input(var, Distribution::uniform_categorical(num_cats));
        let mut prod_children = vec![input];

        for &child_var in &children_map[var] {
            let child_subtree =
                build_hclt_recursive(b, child_var, children_map, num_cats, num_latents, data);
            prod_children.push(child_subtree);
        }

        let prod = b.add_product(prod_children);
        components.push(prod);
    }

    // Sum over components (uniform initial weights)
    b.add_sum(components)
}

/// Compute empirical distribution from data for a variable.
fn empirical_distribution(data: &[Vec<usize>], var: usize, num_cats: usize) -> Distribution {
    let mut counts = vec![0.0f64; num_cats];
    for sample in data {
        let v = sample[var].min(num_cats - 1);
        counts[v] += 1.0;
    }
    Distribution::categorical_from_probs(&counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive() {
        let b = naive(3, 4);
        assert!(b.len() >= 3); // at least 3 input nodes
        assert_eq!(b.num_vars(), 3);
    }

    #[test]
    fn test_mutual_information() {
        // Perfectly correlated: X0 = X1
        let data: Vec<Vec<usize>> = (0..100).map(|i| vec![i % 2, i % 2]).collect();
        let mi = mutual_information(&data, 2);
        assert!(
            mi[0][1] > 0.5,
            "MI should be high for correlated vars: {}",
            mi[0][1]
        );
    }

    #[test]
    fn test_hclt_builds() {
        let data: Vec<Vec<usize>> = (0..50).map(|i| vec![i % 3, (i + 1) % 3, i % 2]).collect();
        let b = hclt(&data, 3, 2);
        assert!(b.len() > 3, "HCLT should have more nodes than variables");
        assert_eq!(b.num_vars(), 3);
    }
}
