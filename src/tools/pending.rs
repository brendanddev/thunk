#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Describes a tool action that requires explicit user approval before execution.
/// Pure data — no methods, no logic. Owned by the runtime between proposal and resolution.
#[derive(Debug, Clone)]
pub struct PendingAction {
    pub tool_name: String,
    pub summary: String,
    pub risk: RiskLevel,
    /// Opaque serialized payload passed back to the tool's execute_approved().
    pub payload: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_action_is_cloneable() {
        let action = PendingAction {
            tool_name: "edit_file".to_string(),
            summary: "Edit src/lib.rs (3 lines)".to_string(),
            risk: RiskLevel::Low,
            payload: "{}".to_string(),
        };
        let cloned = action.clone();
        assert_eq!(cloned.tool_name, "edit_file");
        assert_eq!(cloned.risk, RiskLevel::Low);
        assert_eq!(cloned.summary, action.summary);
    }

    #[test]
    fn risk_levels_are_comparable() {
        assert_eq!(RiskLevel::Low, RiskLevel::Low);
        assert_ne!(RiskLevel::Low, RiskLevel::High);
        assert_ne!(RiskLevel::Medium, RiskLevel::High);
    }
}
