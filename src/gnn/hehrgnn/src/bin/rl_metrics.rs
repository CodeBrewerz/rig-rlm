use hehrgnn::eval::agent_env::AgentEnv;
use hehrgnn::eval::environment::Environment;
use hehrgnn::eval::fiduciary_env::FiduciaryEnv;
use hehrgnn::eval::rl_policy::{EpsilonGreedyPolicy, RandomPolicy, RuleBasedPolicy};
use hehrgnn::eval::simulator::{run_batch, SimulationReport};
use std::fs::File;
use std::io::Write;

/// Generate empirical metrics for RL Environments
fn main() -> std::io::Result<()> {
    println!("Gathering empirical metrics for RL Environments...");

    // Create CSV file for agent env
    let mut file = File::create("rl_agent_metrics.csv")?;
    writeln!(file, "environment,policy,episodes,mean_reward,std_reward,mean_steps,mean_constraint_cost,violation_count")?;

    // Agent Environment Test
    let mut env = AgentEnv::new();
    let num_episodes = 1000;

    // 1. Random Policy
    let mut p_rand = RandomPolicy::new(42);
    let report_rand = run_batch(&mut env, &mut p_rand, num_episodes, None, 50.0);
    write_report(&mut file, "AgentEnv", &report_rand)?;

    // 2. RuleBased Policy
    let mut p_rule = RuleBasedPolicy::active_agent();
    let report_rule = run_batch(&mut env, &mut p_rule, num_episodes, None, 50.0);
    write_report(&mut file, "AgentEnv", &report_rule)?;

    // 3. Epsilon Greedy Policy
    let mut p_eps = EpsilonGreedyPolicy::new(env.available_actions().len(), 0.5, 0.999);
    let report_eps = run_batch(&mut env, &mut p_eps, num_episodes, None, 50.0);
    write_report(&mut file, "AgentEnv", &report_eps)?;

    // Fiduciary Environment Test
    let mut env_fid = FiduciaryEnv::new(4);

    // 1. Random Policy
    let mut p_rand_f = RandomPolicy::new(42);
    let report_rand_f = run_batch(&mut env_fid, &mut p_rand_f, num_episodes, None, 1.0);
    write_report(&mut file, "FiduciaryEnv", &report_rand_f)?;

    // 2. RuleBased Policy (Conservative)
    let mut p_rule_f = RuleBasedPolicy::conservative_fiduciary();
    let report_rule_f = run_batch(&mut env_fid, &mut p_rule_f, num_episodes, None, 1.0);
    write_report(&mut file, "FiduciaryEnv", &report_rule_f)?;

    println!("Empirical metrics successfully exported to 'rl_agent_metrics.csv' ({} total episodes per policy).", num_episodes);
    Ok(())
}

fn write_report(file: &mut File, env_name: &str, report: &SimulationReport) -> std::io::Result<()> {
    writeln!(
        file,
        "{},{},{},{:.4},{:.4},{:.2},{:.4},{}",
        env_name,
        report.policy_name,
        report.num_episodes,
        report.mean_reward,
        report.std_reward,
        report.mean_steps,
        report.mean_constraint_cost,
        report.violation_count
    )
}
