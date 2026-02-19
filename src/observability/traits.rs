use std::time::Duration;

/// Events the observer can record
#[derive(Debug, Clone)]
pub enum ObserverEvent {
    AgentStart {
        provider: String,
        model: String,
    },
    /// A request is about to be sent to an LLM provider.
    ///
    /// This is emitted immediately before a provider call so observers can print
    /// user-facing progress without leaking prompt contents.
    LlmRequest {
        provider: String,
        model: String,
        messages_count: usize,
    },
    /// Result of a single LLM provider call.
    LlmResponse {
        provider: String,
        model: String,
        duration: Duration,
        success: bool,
        error_message: Option<String>,
    },
    AgentEnd {
        provider: String,
        model: String,
        duration: Duration,
        tokens_used: Option<u64>,
        cost_usd: Option<f64>,
    },
    /// A tool call is about to be executed.
    ToolCallStart {
        tool: String,
    },
    ToolCall {
        tool: String,
        duration: Duration,
        success: bool,
    },
    /// The agent produced a final answer for the current user message.
    TurnComplete,
    ChannelMessage {
        channel: String,
        direction: String,
    },
    HeartbeatTick,
    Error {
        component: String,
        message: String,
    },
}

/// Numeric metrics
#[derive(Debug, Clone)]
pub enum ObserverMetric {
    RequestLatency(Duration),
    TokensUsed(u64),
    ActiveSessions(u64),
    QueueDepth(u64),
}

/// Core observability trait — implement for any backend
pub trait Observer: Send + Sync + 'static {
    /// Record a discrete event
    fn record_event(&self, event: &ObserverEvent);

    /// Record a numeric metric
    fn record_metric(&self, metric: &ObserverMetric);

    /// Flush any buffered data (no-op for most backends)
    fn flush(&self) {}

    /// Human-readable name of this observer
    fn name(&self) -> &str;

    /// Downcast to `Any` for backend-specific operations
    fn as_any(&self) -> &dyn std::any::Any;
}

// ── Intervention Handler (AutoGen InterventionHandler analog) ──────────

/// Verdict returned by an intervention handler.
#[derive(Debug, Clone)]
pub enum InterventionVerdict {
    /// Allow the message through unchanged.
    Allow,
    /// Replace the message content with the provided string.
    Modify(String),
    /// Drop the message entirely. Contains a reason string.
    Drop(String),
    /// Halt the agent loop entirely. Critical safety violation.
    Halt(String),
}

/// Message direction for intervention context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageDirection {
    Inbound,
    OutboundRequest,
    InboundResponse,
    ToolInvocation,
    ToolResult,
    OutboundResponse,
}

/// Context provided to intervention handlers.
#[derive(Debug, Clone)]
pub struct InterventionContext {
    pub direction: MessageDirection,
    pub agent_id: Option<String>,
    pub tool_name: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Message-level middleware that can intercept, modify, or drop messages.
/// ZeroClaw analog of AutoGen's InterventionHandler.
pub trait InterventionHandler: Send + Sync + 'static {
    fn intercept(&self, content: &str, ctx: &InterventionContext) -> InterventionVerdict;
    fn name(&self) -> &str;
}

/// No-op handler that allows everything.
pub struct NoopInterventionHandler;

impl InterventionHandler for NoopInterventionHandler {
    fn intercept(&self, _content: &str, _ctx: &InterventionContext) -> InterventionVerdict {
        InterventionVerdict::Allow
    }
    fn name(&self) -> &str { "noop" }
}

/// Chain of handlers. First Drop/Halt wins.
pub struct InterventionChain {
    handlers: Vec<Box<dyn InterventionHandler>>,
}

impl InterventionChain {
    pub fn new() -> Self { Self { handlers: Vec::new() } }

    pub fn add(&mut self, handler: Box<dyn InterventionHandler>) {
        self.handlers.push(handler);
    }

    pub fn process(&self, content: &str, ctx: &InterventionContext) -> InterventionVerdict {
        let mut current = content.to_string();
        for handler in &self.handlers {
            match handler.intercept(&current, ctx) {
                InterventionVerdict::Allow => continue,
                InterventionVerdict::Modify(new) => {
                    tracing::debug!(handler = handler.name(), "InterventionHandler modified message");
                    current = new;
                }
                InterventionVerdict::Drop(reason) => {
                    tracing::warn!(handler = handler.name(), reason = %reason, "InterventionHandler dropped message");
                    return InterventionVerdict::Drop(reason);
                }
                InterventionVerdict::Halt(reason) => {
                    tracing::error!(handler = handler.name(), reason = %reason, "InterventionHandler HALT");
                    return InterventionVerdict::Halt(reason);
                }
            }
        }
        if current != content { InterventionVerdict::Modify(current) } else { InterventionVerdict::Allow }
    }

    pub fn is_empty(&self) -> bool { self.handlers.is_empty() }
}

impl Default for InterventionChain {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::time::Duration;

    #[derive(Default)]
    struct DummyObserver {
        events: Mutex<u64>,
        metrics: Mutex<u64>,
    }

    impl Observer for DummyObserver {
        fn record_event(&self, _event: &ObserverEvent) {
            let mut guard = self.events.lock();
            *guard += 1;
        }

        fn record_metric(&self, _metric: &ObserverMetric) {
            let mut guard = self.metrics.lock();
            *guard += 1;
        }

        fn name(&self) -> &str {
            "dummy-observer"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn observer_records_events_and_metrics() {
        let observer = DummyObserver::default();

        observer.record_event(&ObserverEvent::HeartbeatTick);
        observer.record_event(&ObserverEvent::Error {
            component: "test".into(),
            message: "boom".into(),
        });
        observer.record_metric(&ObserverMetric::TokensUsed(42));

        assert_eq!(*observer.events.lock(), 2);
        assert_eq!(*observer.metrics.lock(), 1);
    }

    #[test]
    fn observer_default_flush_and_as_any_work() {
        let observer = DummyObserver::default();

        observer.flush();
        assert_eq!(observer.name(), "dummy-observer");
        assert!(observer.as_any().downcast_ref::<DummyObserver>().is_some());
    }

    #[test]
    fn observer_event_and_metric_are_cloneable() {
        let event = ObserverEvent::ToolCall {
            tool: "shell".into(),
            duration: Duration::from_millis(10),
            success: true,
        };
        let metric = ObserverMetric::RequestLatency(Duration::from_millis(8));

        let cloned_event = event.clone();
        let cloned_metric = metric.clone();

        assert!(matches!(cloned_event, ObserverEvent::ToolCall { .. }));
        assert!(matches!(cloned_metric, ObserverMetric::RequestLatency(_)));
    }

    // ── InterventionHandler tests ──

    struct AllowH;
    impl InterventionHandler for AllowH {
        fn intercept(&self, _c: &str, _x: &InterventionContext) -> InterventionVerdict { InterventionVerdict::Allow }
        fn name(&self) -> &str { "allow" }
    }

    struct DropH(String);
    impl InterventionHandler for DropH {
        fn intercept(&self, _c: &str, _x: &InterventionContext) -> InterventionVerdict { InterventionVerdict::Drop(self.0.clone()) }
        fn name(&self) -> &str { "drop" }
    }

    struct UpperH;
    impl InterventionHandler for UpperH {
        fn intercept(&self, c: &str, _x: &InterventionContext) -> InterventionVerdict { InterventionVerdict::Modify(c.to_uppercase()) }
        fn name(&self) -> &str { "upper" }
    }

    fn ictx() -> InterventionContext {
        InterventionContext { direction: MessageDirection::Inbound, agent_id: None, tool_name: None, provider: None, model: None }
    }

    #[test]
    fn noop_handler_allows() {
        assert!(matches!(NoopInterventionHandler.intercept("x", &ictx()), InterventionVerdict::Allow));
    }

    #[test]
    fn chain_empty_allows() {
        assert!(matches!(InterventionChain::new().process("x", &ictx()), InterventionVerdict::Allow));
    }

    #[test]
    fn chain_drop_stops() {
        let mut c = InterventionChain::new();
        c.add(Box::new(AllowH));
        c.add(Box::new(DropH("no".into())));
        c.add(Box::new(AllowH));
        assert!(matches!(c.process("x", &ictx()), InterventionVerdict::Drop(r) if r == "no"));
    }

    #[test]
    fn chain_modify_feeds_forward() {
        let mut c = InterventionChain::new();
        c.add(Box::new(UpperH));
        assert!(matches!(c.process("hi", &ictx()), InterventionVerdict::Modify(ref s) if s == "HI"));
    }
}
