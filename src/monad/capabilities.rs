//! Phase 11: Capability-based tool segregation for agents.
//!
//! Controls which monadic actions each agent is allowed to perform.
//! Root agents get full capabilities; sub-agents get restricted sets
//! to enforce the principle of least privilege.
//!
//! Usage:
//! ```no_run
//! use rig_rlm::monad::Capabilities;
//! let root_caps = Capabilities::root();          // full access
//! let worker = Capabilities::code_worker();      // code + inference, no sub-agents
//! let analyst = Capabilities::analyst();         // inference only, no code
//! ```

use std::fmt;

/// Capability flags controlling which actions an agent may perform.
///
/// Each flag gates a specific `Action` variant in `interpret_action()`.
/// If a capability is `false` and the agent tries to use it, the
/// interpreter returns `AgentError::PermissionDenied`.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Can run code in the sandbox (`Action::ExecuteCode`).
    pub code_execution: bool,
    /// Can call the LLM (`Action::ModelInference`).
    pub model_inference: bool,
    /// Can use `llm_query()` from within sandbox code.
    pub sub_llm_bridge: bool,
    /// Can run shell commands (via subprocess in ExecuteCode).
    pub shell_commands: bool,
    /// Can store/retrieve context variables (`Action::Capture` / `Action::Retrieve`).
    pub variable_capture: bool,
    /// Can spawn child agents (`Action::SpawnSubAgent`).
    pub spawn_sub_agents: bool,
}

impl Capabilities {
    /// Root agent: full access to all capabilities.
    pub fn root() -> Self {
        Self {
            code_execution: true,
            model_inference: true,
            sub_llm_bridge: true,
            shell_commands: true,
            variable_capture: true,
            spawn_sub_agents: true,
        }
    }

    /// Code worker sub-agent: code execution + inference, but no sub-agent spawning.
    ///
    /// Use this for sub-agents that need to analyze/execute code chunks
    /// but shouldn't create further sub-agents (prevents recursive spawning).
    pub fn code_worker() -> Self {
        Self {
            code_execution: true,
            model_inference: true,
            sub_llm_bridge: false,
            shell_commands: false,
            variable_capture: true,
            spawn_sub_agents: false,
        }
    }

    /// Analysis sub-agent: inference only, no code execution.
    ///
    /// Use this for sub-agents that reason about results or provide
    /// feedback without touching the sandbox.
    pub fn analyst() -> Self {
        Self {
            code_execution: false,
            model_inference: true,
            sub_llm_bridge: false,
            shell_commands: false,
            variable_capture: true,
            spawn_sub_agents: false,
        }
    }

    /// Minimal: inference only, nothing else.
    pub fn inference_only() -> Self {
        Self {
            code_execution: false,
            model_inference: true,
            sub_llm_bridge: false,
            shell_commands: false,
            variable_capture: false,
            spawn_sub_agents: false,
        }
    }

    /// Check if a specific action type is allowed by these capabilities.
    ///
    /// Returns `Ok(())` if allowed, `Err(reason)` if denied.
    pub fn check_action(&self, action_name: &str) -> Result<(), String> {
        match action_name {
            "ExecuteCode" => {
                if self.code_execution {
                    Ok(())
                } else {
                    Err("code execution not permitted for this agent".into())
                }
            }
            "ModelInference" => {
                if self.model_inference {
                    Ok(())
                } else {
                    Err("model inference not permitted for this agent".into())
                }
            }
            "ShellCommand" => {
                if self.shell_commands {
                    Ok(())
                } else {
                    Err("shell commands not permitted for this agent".into())
                }
            }
            "SpawnSubAgent" => {
                if self.spawn_sub_agents {
                    Ok(())
                } else {
                    Err("sub-agent spawning not permitted for this agent".into())
                }
            }
            "Capture" | "Retrieve" => {
                if self.variable_capture {
                    Ok(())
                } else {
                    Err("variable capture not permitted for this agent".into())
                }
            }
            // Insert and Log are always allowed
            _ => Ok(()),
        }
    }
}

impl Default for Capabilities {
    /// Default capabilities = root (full access).
    fn default() -> Self {
        Self::root()
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut caps = Vec::new();
        if self.code_execution {
            caps.push("code");
        }
        if self.model_inference {
            caps.push("inference");
        }
        if self.sub_llm_bridge {
            caps.push("sub-llm");
        }
        if self.shell_commands {
            caps.push("shell");
        }
        if self.variable_capture {
            caps.push("vars");
        }
        if self.spawn_sub_agents {
            caps.push("spawn");
        }
        write!(f, "[{}]", caps.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_allows_everything() {
        let caps = Capabilities::root();
        assert!(caps.check_action("ExecuteCode").is_ok());
        assert!(caps.check_action("ModelInference").is_ok());
        assert!(caps.check_action("ShellCommand").is_ok());
        assert!(caps.check_action("SpawnSubAgent").is_ok());
        assert!(caps.check_action("Capture").is_ok());
        assert!(caps.check_action("Insert").is_ok());
    }

    #[test]
    fn test_analyst_denies_code() {
        let caps = Capabilities::analyst();
        assert!(caps.check_action("ExecuteCode").is_err());
        assert!(caps.check_action("ModelInference").is_ok());
        assert!(caps.check_action("SpawnSubAgent").is_err());
        assert!(caps.check_action("ShellCommand").is_err());
    }

    #[test]
    fn test_code_worker_denies_spawn() {
        let caps = Capabilities::code_worker();
        assert!(caps.check_action("ExecuteCode").is_ok());
        assert!(caps.check_action("ModelInference").is_ok());
        assert!(caps.check_action("SpawnSubAgent").is_err());
        assert!(caps.check_action("ShellCommand").is_err());
    }

    #[test]
    fn test_inference_only_denies_variables() {
        let caps = Capabilities::inference_only();
        assert!(caps.check_action("Capture").is_err());
        assert!(caps.check_action("Retrieve").is_err());
        assert!(caps.check_action("ModelInference").is_ok());
    }

    #[test]
    fn test_display() {
        let caps = Capabilities::code_worker();
        let display = format!("{caps}");
        assert!(display.contains("code"));
        assert!(display.contains("inference"));
        assert!(!display.contains("spawn"));
    }
}
