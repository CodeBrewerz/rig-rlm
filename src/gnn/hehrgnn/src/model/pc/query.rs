//! High-level query API for Probabilistic Circuits.
//!
//! - `marginal(circuit, evidence)` → log P(evidence)
//! - `conditional(circuit, evidence, targets)` → P(target_var = x | evidence) for all x
//! - `sample(circuit, evidence, n)` → n samples from P(missing | evidence)

use std::collections::HashMap;

use super::circuit::{CompiledCircuit, Evidence};
use super::distribution::Distribution;
use super::node::NodeKind;

/// Compute log P(evidence), marginalizing over unobserved variables.
///
/// `evidence[var] = Some(x)` for observed variables, `None` for marginalized.
pub fn marginal(circuit: &mut CompiledCircuit, evidence: &Evidence) -> f64 {
    circuit.forward(evidence)
}

/// Compute conditional probabilities P(target_var = x | evidence) for each
/// value x of each target variable.
///
/// Returns: HashMap<var, Vec<f64>> where Vec[x] = P(var=x | evidence).
pub fn conditional(
    circuit: &mut CompiledCircuit,
    evidence: &Evidence,
    target_vars: &[usize],
) -> HashMap<usize, Vec<f64>> {
    // First, compute log P(evidence) with target vars marginalized
    let mut marg_evidence = evidence.clone();
    for &var in target_vars {
        if var < marg_evidence.len() {
            marg_evidence[var] = None;
        }
    }
    let log_p_evidence = circuit.forward(&marg_evidence);

    let mut result = HashMap::new();

    for &var in target_vars {
        // Determine number of categories for this variable
        let num_cats = find_var_categories(circuit, var);

        // If evidence is unseen (P=0), fall back to uniform
        if log_p_evidence == f64::NEG_INFINITY || !log_p_evidence.is_finite() {
            let uniform = 1.0 / num_cats as f64;
            result.insert(var, vec![uniform; num_cats]);
            continue;
        }

        let mut probs = vec![0.0f64; num_cats];

        for x in 0..num_cats {
            // Set target var to x, compute log P(evidence, var=x)
            let mut ev = evidence.clone();
            if var >= ev.len() {
                ev.resize(var + 1, None);
            }
            ev[var] = Some(x);
            let log_p_joint = circuit.forward(&ev);
            // P(var=x | evidence) = P(evidence, var=x) / P(evidence)
            let p = (log_p_joint - log_p_evidence).exp();
            probs[x] = if p.is_finite() { p } else { 0.0 };
        }

        // Normalize (for numerical safety)
        let sum: f64 = probs.iter().sum();
        if sum > 1e-15 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        } else {
            // All zero — uniform fallback
            let uniform = 1.0 / num_cats as f64;
            probs = vec![uniform; num_cats];
        }

        result.insert(var, probs);
    }

    result
}

/// Sample from the circuit distribution, conditioned on evidence.
///
/// Returns `n` samples, each a vector of variable assignments.
/// Observed variables keep their evidence values; unobserved are sampled.
pub fn sample(circuit: &mut CompiledCircuit, evidence: &Evidence, n: usize) -> Vec<Vec<usize>> {
    let num_vars = circuit.num_vars;
    let mut samples = Vec::with_capacity(n);

    // Find all unobserved variables
    let unobserved: Vec<usize> = (0..num_vars)
        .filter(|v| evidence.get(*v).map_or(true, |e| e.is_none()))
        .collect();

    for _ in 0..n {
        let mut assignment = vec![0usize; num_vars];

        // Copy evidence
        for (v, val) in evidence.iter().enumerate() {
            if let Some(x) = val {
                assignment[v] = *x;
            }
        }

        // Sample unobserved variables one by one (forward-filter)
        for &var in &unobserved {
            let num_cats = find_var_categories(circuit, var);
            if num_cats == 0 {
                continue;
            }

            // Compute P(var=x | evidence, already-sampled vars)
            let mut ev: Evidence = assignment.iter().map(|&v| Some(v)).collect();
            // Marginalize this var
            ev[var] = None;
            let log_p_marg = circuit.forward(&ev);

            let mut probs = Vec::with_capacity(num_cats);
            for x in 0..num_cats {
                ev[var] = Some(x);
                let log_p = circuit.forward(&ev);
                probs.push((log_p - log_p_marg).exp());
            }

            // Normalize
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }

            // Sample from categorical
            let u: f64 = simple_random(var as u64);
            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (x, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u < cumsum {
                    chosen = x;
                    break;
                }
            }
            assignment[var] = chosen;
        }

        samples.push(assignment);
    }

    samples
}

/// Find the number of categories for a variable by looking at input nodes.
fn find_var_categories(circuit: &CompiledCircuit, var: usize) -> usize {
    for node in &circuit.nodes {
        if let NodeKind::Input { var: v, dist } = &node.kind {
            if *v == var {
                return dist.num_categories();
            }
        }
    }
    2 // default to binary
}

/// Simple deterministic pseudo-random for sampling (no rand crate dependency).
fn simple_random(seed: u64) -> f64 {
    // Splitmix64
    static mut COUNTER: u64 = 0;
    let s = unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        seed.wrapping_add(COUNTER).wrapping_mul(0x9E3779B97F4A7C15)
    };
    let s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
    let s = s ^ (s >> 31);
    (s >> 11) as f64 / (1u64 << 53) as f64
}

/// Compute the lift (conditional vs marginal ratio) of variable `var` having
/// value `val` on a target event.
///
/// Lift = P(target | var=val) / P(target) = P(var=val, target) / (P(var=val) × P(target))
pub fn lift(
    circuit: &mut CompiledCircuit,
    evidence: &Evidence,
    var: usize,
    val: usize,
    target_var: usize,
    target_val: usize,
) -> f64 {
    let num_vars = circuit.num_vars;

    // P(target)
    let mut ev_target_only: Evidence = vec![None; num_vars];
    ev_target_only[target_var] = Some(target_val);
    let log_p_target = circuit.forward(&ev_target_only);

    // P(target | var=val) via Bayes: P(target, var=val) / P(var=val)
    let mut ev_both: Evidence = vec![None; num_vars];
    ev_both[var] = Some(val);
    ev_both[target_var] = Some(target_val);
    let log_p_both = circuit.forward(&ev_both);

    let mut ev_var_only: Evidence = vec![None; num_vars];
    ev_var_only[var] = Some(val);
    let log_p_var = circuit.forward(&ev_var_only);

    let log_p_target_given_var = log_p_both - log_p_var;

    // Lift = P(target | var) / P(target)
    let lift_val = (log_p_target_given_var - log_p_target).exp();
    if lift_val.is_finite() {
        lift_val
    } else {
        1.0
    } // neutral if unseen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pc::circuit::CompiledCircuit;
    use crate::model::pc::distribution::Distribution;
    use crate::model::pc::node::CircuitBuilder;

    #[test]
    fn test_marginal_query() {
        let mut b = CircuitBuilder::new();
        let x0 = b.add_input(0, Distribution::bernoulli(0.3));
        let x1 = b.add_input(1, Distribution::bernoulli(0.7));
        b.add_product(vec![x0, x1]);

        let mut c = CompiledCircuit::compile(&b);

        // P(X0=1) with X1 marginalized = 0.3
        let ll = marginal(&mut c, &vec![Some(1), None]);
        assert!((ll.exp() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_conditional_query() {
        // P(X1 | X0) where X0 and X1 are independent
        let mut b = CircuitBuilder::new();
        let x0 = b.add_input(0, Distribution::bernoulli(0.3));
        let x1 = b.add_input(1, Distribution::bernoulli(0.7));
        b.add_product(vec![x0, x1]);

        let mut c = CompiledCircuit::compile(&b);

        // P(X1=1 | X0=0) should = P(X1=1) = 0.7 (independent)
        let cond = conditional(&mut c, &vec![Some(0), None], &[1]);
        let p_x1 = cond.get(&1).unwrap();
        assert!(
            (p_x1[1] - 0.7).abs() < 1e-4,
            "P(X1=1|X0=0) = {} (expected 0.7)",
            p_x1[1]
        );
    }

    #[test]
    fn test_conditional_mixture() {
        // Correlated mixture:
        // Component A (w=0.6): X0~Bern(0.9), X1~Bern(0.1)
        // Component B (w=0.4): X0~Bern(0.1), X1~Bern(0.9)
        // P(X1=1 | X0=1) should be biased toward 0.1 (component A dominates)
        let mut b = CircuitBuilder::new();
        let x0a = b.add_input(0, Distribution::bernoulli(0.9));
        let x1a = b.add_input(1, Distribution::bernoulli(0.1));
        let pa = b.add_product(vec![x0a, x1a]);

        let x0b = b.add_input(0, Distribution::bernoulli(0.1));
        let x1b = b.add_input(1, Distribution::bernoulli(0.9));
        let pb = b.add_product(vec![x0b, x1b]);

        b.add_sum_weighted(vec![pa, pb], vec![0.6_f64.ln(), 0.4_f64.ln()]);

        let mut c = CompiledCircuit::compile(&b);

        let cond = conditional(&mut c, &vec![Some(1), None], &[1]);
        let p_x1 = cond.get(&1).unwrap();

        // When X0=1: component A (P(X0=1)=0.9, w=0.6) dominates
        // P(X1=1|X0=1) ≈ 0.6*0.9*0.1 + 0.4*0.1*0.9 / (0.6*0.9 + 0.4*0.1)
        //              = (0.054 + 0.036) / (0.54 + 0.04) = 0.09/0.58 ≈ 0.155
        println!(
            "P(X1=0|X0=1) = {:.4}, P(X1=1|X0=1) = {:.4}",
            p_x1[0], p_x1[1]
        );
        assert!(p_x1[1] < 0.3, "X1 should be biased low when X0=1");
        assert!((p_x1[0] + p_x1[1] - 1.0).abs() < 1e-6, "Should sum to 1");
    }

    #[test]
    fn test_lift() {
        // Same correlated mixture as above
        let mut b = CircuitBuilder::new();
        let x0a = b.add_input(0, Distribution::bernoulli(0.9));
        let x1a = b.add_input(1, Distribution::bernoulli(0.1));
        let pa = b.add_product(vec![x0a, x1a]);

        let x0b = b.add_input(0, Distribution::bernoulli(0.1));
        let x1b = b.add_input(1, Distribution::bernoulli(0.9));
        let pb = b.add_product(vec![x0b, x1b]);

        b.add_sum_weighted(vec![pa, pb], vec![0.6_f64.ln(), 0.4_f64.ln()]);

        let mut c = CompiledCircuit::compile(&b);

        // Lift of X0=1 on X1=1
        let l = lift(&mut c, &vec![None, None], 0, 1, 1, 1);
        println!("Lift(X0=1 → X1=1) = {:.4}", l);
        // X0=1 should decrease X1=1 probability (negative correlation in mixture)
        assert!(l < 1.0, "Lift should be < 1 (negative correlation)");
    }
}
