//! Built-in InterventionHandler implementations.
//!
//! These handlers plug into the InterventionChain to enforce behavioral
//! constraints at the message level. They are compiled into the binary
//! and cannot be bypassed by prompt injection.

use crate::observability::{
    InterventionContext, InterventionHandler, InterventionVerdict, MessageDirection,
};
use regex::Regex;
use std::sync::atomic::{AtomicU32, Ordering};

// ── TripwireHandler: regex-based halt on forbidden content ────────────────────

/// Checks all message content against a set of compiled regex patterns.
/// If any pattern matches, returns Halt immediately. This is the runtime
/// equivalent of a circuit breaker — it cannot be reasoned with or talked out of.
pub struct TripwireHandler {
    patterns: Vec<Regex>,
}

impl TripwireHandler {
    pub fn new(patterns: Vec<Regex>) -> Self {
        Self { patterns }
    }

    /// Build from string patterns, skipping any that fail to compile.
    pub fn from_strings(patterns: &[String]) -> Self {
        let compiled = patterns
            .iter()
            .filter_map(|p| match Regex::new(p) {
                Ok(re) => Some(re),
                Err(e) => {
                    tracing::warn!("TripwireHandler: invalid pattern '{}': {}", p, e);
                    None
                }
            })
            .collect();
        Self { patterns: compiled }
    }
}

impl InterventionHandler for TripwireHandler {
    fn intercept(&self, content: &str, _ctx: &InterventionContext) -> InterventionVerdict {
        for pattern in &self.patterns {
            if pattern.is_match(content) {
                return InterventionVerdict::Halt(format!(
                    "TRIPWIRE: content matched forbidden pattern /{}/",
                    pattern.as_str()
                ));
            }
        }
        InterventionVerdict::Allow
    }

    fn name(&self) -> &str {
        "tripwire"
    }
}

// ── SingleActionHandler: enforce one tool call per turn ──────────────────────

/// Tracks tool invocations per turn. If more than one tool call is attempted
/// in a single turn, subsequent calls are dropped. The counter must be reset
/// between turns by calling `reset()`.
///
/// This enforces the ACT discipline: one action per turn, verify before next.
pub struct SingleActionHandler {
    calls_this_turn: AtomicU32,
    max_per_turn: u32,
}

impl SingleActionHandler {
    pub fn new(max_per_turn: u32) -> Self {
        Self {
            calls_this_turn: AtomicU32::new(0),
            max_per_turn,
        }
    }

    /// Reset the counter at the start of each turn.
    pub fn reset(&self) {
        self.calls_this_turn.store(0, Ordering::SeqCst);
    }

    /// Get current count (for testing/observability).
    pub fn count(&self) -> u32 {
        self.calls_this_turn.load(Ordering::SeqCst)
    }
}

impl InterventionHandler for SingleActionHandler {
    fn intercept(&self, _content: &str, ctx: &InterventionContext) -> InterventionVerdict {
        // Only enforce on tool invocations
        if ctx.direction != MessageDirection::ToolInvocation {
            return InterventionVerdict::Allow;
        }

        let count = self.calls_this_turn.fetch_add(1, Ordering::SeqCst);
        if count >= self.max_per_turn {
            InterventionVerdict::Drop(format!(
                "ACT discipline: max {} tool call(s) per turn exceeded (attempted #{})",
                self.max_per_turn,
                count + 1
            ))
        } else {
            InterventionVerdict::Allow
        }
    }

    fn name(&self) -> &str {
        "single-action"
    }
}

// ── DepthGuardHandler: block delegation from delegates ───────────────────────

/// Prevents sub-agents from spawning further sub-agents. If the current agent
/// already has an agent_id (meaning it IS a delegate), any tool invocation
/// of "delegate" is dropped. This enforces one-depth dispatch.
pub struct DepthGuardHandler;

impl InterventionHandler for DepthGuardHandler {
    fn intercept(&self, _content: &str, ctx: &InterventionContext) -> InterventionVerdict {
        // Only enforce on tool invocations
        if ctx.direction != MessageDirection::ToolInvocation {
            return InterventionVerdict::Allow;
        }

        // If this agent has an identity (it's a delegate) and is trying to delegate...
        if let Some(ref agent_id) = ctx.agent_id {
            if let Some(ref tool) = ctx.tool_name {
                if tool == "delegate" {
                    return InterventionVerdict::Drop(format!(
                        "One-depth dispatch: agent '{}' cannot delegate further",
                        agent_id
                    ));
                }
            }
        }

        InterventionVerdict::Allow
    }

    fn name(&self) -> &str {
        "depth-guard"
    }
}

// ── ConvergenceDetector: flag when delegate outputs are too similar ───────────

/// Tracks delegate response outputs and flags when they converge.
/// This is not a blocking handler — it modifies the response to prepend
/// a convergence warning when similarity exceeds the threshold.
///
/// Similarity is measured by Jaccard index on word-level trigrams.
/// This is fast, allocation-light, and good enough for detecting
/// copy-paste agreement vs genuinely independent conclusions.
pub struct ConvergenceDetector {
    /// Jaccard similarity threshold (0.0 - 1.0). Above this = convergence warning.
    threshold: f64,
    /// Collected delegate outputs for the current evaluation round.
    outputs: parking_lot::Mutex<Vec<String>>,
}

impl ConvergenceDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            outputs: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Reset collected outputs between evaluation rounds.
    pub fn reset(&self) {
        self.outputs.lock().clear();
    }

    /// Compute Jaccard similarity between two strings using word trigrams.
    fn jaccard_trigrams(a: &str, b: &str) -> f64 {
        use std::collections::HashSet;

        let trigrams = |s: &str| -> HashSet<String> {
            let words: Vec<&str> = s.split_whitespace().collect();
            if words.len() < 3 {
                return words.iter().map(|w| w.to_string()).collect();
            }
            words.windows(3).map(|w| w.join(" ")).collect()
        };

        let a_set = trigrams(a);
        let b_set = trigrams(b);

        if a_set.is_empty() && b_set.is_empty() {
            return 1.0; // both empty = identical
        }

        let intersection = a_set.intersection(&b_set).count();
        let union = a_set.union(&b_set).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f64 / union as f64
    }

    /// Check if the new output converges with any stored output.
    fn check_convergence(&self, new_output: &str) -> Option<f64> {
        let outputs = self.outputs.lock();
        for existing in outputs.iter() {
            let similarity = Self::jaccard_trigrams(existing, new_output);
            if similarity >= self.threshold {
                return Some(similarity);
            }
        }
        None
    }
}

impl InterventionHandler for ConvergenceDetector {
    fn intercept(&self, content: &str, ctx: &InterventionContext) -> InterventionVerdict {
        // Only check tool results (delegate responses coming back)
        if ctx.direction != MessageDirection::ToolResult {
            return InterventionVerdict::Allow;
        }

        // Only care about delegate tool results
        if ctx.tool_name.as_deref() != Some("delegate") {
            return InterventionVerdict::Allow;
        }

        if let Some(similarity) = self.check_convergence(content) {
            tracing::warn!(
                similarity = %format!("{:.1}%", similarity * 100.0),
                "Convergence detected in delegate responses"
            );
            // Don't block — modify to prepend warning
            let warning = format!(
                "[CONVERGENCE WARNING: This delegate response has {:.0}% similarity with a prior \
                 delegate response. Unanimous agreement is a failure signal. Re-examine with \
                 explicit skepticism.]\n\n{}",
                similarity * 100.0,
                content
            );
            self.outputs.lock().push(content.to_string());
            InterventionVerdict::Modify(warning)
        } else {
            self.outputs.lock().push(content.to_string());
            InterventionVerdict::Allow
        }
    }

    fn name(&self) -> &str {
        "convergence-detector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::InterventionContext;

    fn tool_ctx(agent_id: Option<&str>, tool: &str) -> InterventionContext {
        InterventionContext {
            direction: MessageDirection::ToolInvocation,
            agent_id: agent_id.map(|s| s.to_string()),
            tool_name: Some(tool.to_string()),
            provider: None,
            model: None,
        }
    }

    fn inbound_ctx() -> InterventionContext {
        InterventionContext {
            direction: MessageDirection::Inbound,
            agent_id: None,
            tool_name: None,
            provider: None,
            model: None,
        }
    }

    fn delegate_result_ctx() -> InterventionContext {
        InterventionContext {
            direction: MessageDirection::ToolResult,
            agent_id: None,
            tool_name: Some("delegate".to_string()),
            provider: None,
            model: None,
        }
    }

    // ── TripwireHandler tests ──

    #[test]
    fn tripwire_allows_clean_content() {
        let h = TripwireHandler::from_strings(&["(?i)rm\\s+-rf\\s+/".into()]);
        let v = h.intercept("hello world", &inbound_ctx());
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn tripwire_halts_on_match() {
        let h = TripwireHandler::from_strings(&["(?i)rm\\s+-rf\\s+/".into()]);
        let v = h.intercept("run: rm -rf /", &inbound_ctx());
        assert!(matches!(v, InterventionVerdict::Halt(_)));
    }

    #[test]
    fn tripwire_skips_invalid_patterns() {
        let h = TripwireHandler::from_strings(&["[invalid".into(), "valid".into()]);
        assert_eq!(h.patterns.len(), 1);
    }

    // ── SingleActionHandler tests ──

    #[test]
    fn single_action_allows_first_call() {
        let h = SingleActionHandler::new(1);
        let v = h.intercept("{}", &tool_ctx(None, "file_read"));
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn single_action_drops_second_call() {
        let h = SingleActionHandler::new(1);
        h.intercept("{}", &tool_ctx(None, "file_read"));
        let v = h.intercept("{}", &tool_ctx(None, "file_write"));
        assert!(matches!(v, InterventionVerdict::Drop(_)));
    }

    #[test]
    fn single_action_reset_allows_again() {
        let h = SingleActionHandler::new(1);
        h.intercept("{}", &tool_ctx(None, "file_read"));
        h.reset();
        let v = h.intercept("{}", &tool_ctx(None, "file_write"));
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn single_action_ignores_non_tool_directions() {
        let h = SingleActionHandler::new(1);
        let v = h.intercept("hello", &inbound_ctx());
        assert!(matches!(v, InterventionVerdict::Allow));
        assert_eq!(h.count(), 0);
    }

    // ── DepthGuardHandler tests ──

    #[test]
    fn depth_guard_allows_root_to_delegate() {
        let h = DepthGuardHandler;
        let v = h.intercept("{}", &tool_ctx(None, "delegate"));
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn depth_guard_blocks_delegate_from_delegating() {
        let h = DepthGuardHandler;
        let v = h.intercept("{}", &tool_ctx(Some("worker-1"), "delegate"));
        assert!(matches!(v, InterventionVerdict::Drop(_)));
    }

    #[test]
    fn depth_guard_allows_delegate_other_tools() {
        let h = DepthGuardHandler;
        let v = h.intercept("{}", &tool_ctx(Some("worker-1"), "file_read"));
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn depth_guard_ignores_non_tool_messages() {
        let h = DepthGuardHandler;
        let v = h.intercept("hello", &inbound_ctx());
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    // ── ConvergenceDetector tests ──

    #[test]
    fn first_result_always_allowed() {
        let d = ConvergenceDetector::new(0.7);
        let v = d.intercept("The code looks correct.", &delegate_result_ctx());
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn different_results_allowed() {
        let d = ConvergenceDetector::new(0.7);
        d.intercept(
            "The code has a buffer overflow in line 42.",
            &delegate_result_ctx(),
        );
        let v = d.intercept(
            "Performance could be improved with caching.",
            &delegate_result_ctx(),
        );
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn identical_results_trigger_warning() {
        let d = ConvergenceDetector::new(0.7);
        let msg =
            "The implementation looks correct and follows best practices for error handling.";
        d.intercept(msg, &delegate_result_ctx());
        let v = d.intercept(msg, &delegate_result_ctx());
        assert!(matches!(v, InterventionVerdict::Modify(_)));
    }

    #[test]
    fn reset_clears_history() {
        let d = ConvergenceDetector::new(0.7);
        let msg = "The implementation is correct.";
        d.intercept(msg, &delegate_result_ctx());
        d.reset();
        let v = d.intercept(msg, &delegate_result_ctx());
        assert!(matches!(v, InterventionVerdict::Allow));
    }

    #[test]
    fn jaccard_identical_strings() {
        let sim = ConvergenceDetector::jaccard_trigrams("a b c d e", "a b c d e");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_strings() {
        let sim = ConvergenceDetector::jaccard_trigrams("a b c d e", "f g h i j");
        assert!(sim < 0.01);
    }

    #[test]
    fn non_delegate_results_ignored() {
        let d = ConvergenceDetector::new(0.7);
        let ctx = InterventionContext {
            direction: MessageDirection::ToolResult,
            agent_id: None,
            tool_name: Some("file_read".to_string()),
            provider: None,
            model: None,
        };
        let msg = "same content";
        d.intercept(msg, &ctx);
        let v = d.intercept(msg, &ctx);
        assert!(matches!(v, InterventionVerdict::Allow));
    }
}
