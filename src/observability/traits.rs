use std::time::Duration;

/// Owned snapshot of a single chat message, used by observability events to
/// carry full message bodies into observers without depending on the provider
/// crate's `ChatMessage` type. Constructed via `From<&ChatMessage>` (see
/// `impl From` block below).
#[derive(Debug, Clone)]
pub struct MessageSnapshot {
    pub role: String,
    pub content: String,
}

/// Owned snapshot of a tool definition, used by observability events to
/// carry the tool list registered for an LLM call. Constructed via
/// `From<&ToolSpec>`.
#[derive(Debug, Clone)]
pub struct ToolSpecSnapshot {
    pub name: String,
    pub description: String,
    /// JSON schema for the tool's parameters, serialized as a string so the
    /// observability layer doesn't need a `serde_json` dependency at the
    /// trait boundary. Observers that need the structured form can re-parse.
    pub parameters_json: String,
}

/// Owned snapshot of a tool call requested by the LLM, used by observability
/// events to surface what the model asked the agent to invoke. Constructed
/// via `From<&ToolCall>`.
#[derive(Debug, Clone)]
pub struct ToolCallSnapshot {
    pub id: String,
    pub name: String,
    /// Tool arguments as a JSON string (already serialized by the provider).
    pub arguments_json: String,
}

/// Discrete events emitted by the agent runtime for observability.
///
/// Each variant represents a lifecycle event that observers can record,
/// aggregate, or forward to external monitoring systems. As of Tier 2, LLM
/// events carry full payloads (system prompt, input/output messages, tool
/// definitions, tool call args/results) so OTel exporters can emit the
/// `gen_ai.*` semantic-convention attributes Langfuse and other LLM-aware
/// backends recognize.
#[derive(Debug, Clone)]
pub enum ObserverEvent {
    /// The agent orchestration loop has started a new session.
    AgentStart { provider: String, model: String },
    /// A request is about to be sent to an LLM provider.
    ///
    /// Emitted immediately before a provider call. Stays minimal — observers
    /// that want the full request/response payload should use `LlmResponse`,
    /// which carries both sides of the call so a single span can be built
    /// post-hoc with both input and output attached.
    LlmRequest {
        provider: String,
        model: String,
        messages_count: usize,
    },
    /// Result of a single LLM provider call.
    ///
    /// As of Tier 2, carries the **full** request and response payload so
    /// OTel exporters can build a single span with both `gen_ai.input.*` and
    /// `gen_ai.output.*` semantic-convention attributes. The agent loop
    /// captures the input snapshots immediately before the provider call and
    /// passes them through here alongside the response.
    LlmResponse {
        provider: String,
        model: String,
        duration: Duration,
        success: bool,
        error_message: Option<String>,
        // ── Input side (snapshot taken before the call) ────────────
        /// System prompt content extracted from the conversation history.
        /// `None` when no `role=system` message is present.
        system_prompt: Option<String>,
        /// Full input message snapshots in send order. Includes the system
        /// message if present (so observers see exactly what the provider
        /// received). Empty for callers that don't have a history list
        /// available (e.g. the gateway single-shot path).
        input_messages: Vec<MessageSnapshot>,
        /// Tool definitions registered with this call. Empty when the agent
        /// is in non-tool mode or the caller has no tool registry in scope.
        tool_definitions: Vec<ToolSpecSnapshot>,
        // ── Output side (from the provider response) ───────────────
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        /// Tokens read from the provider's prompt cache (Anthropic
        /// `cache_read_input_tokens`). When the system prompt and tools are
        /// cached, the bulk of the prompt budget shows up here, NOT in
        /// `input_tokens`. Observers should sum all three to get the true
        /// total prompt token count.
        cache_read_input_tokens: Option<u64>,
        /// Tokens written to the provider's prompt cache on this call
        /// (Anthropic `cache_creation_input_tokens`). Non-zero on the first
        /// call that primes the cache; usually zero on subsequent calls.
        cache_creation_input_tokens: Option<u64>,
        /// Assistant text response. May be empty when the model returned only
        /// tool calls. `None` on failure.
        output_text: Option<String>,
        /// Tool calls the model requested in this response. Empty when the
        /// model returned only text or when the call failed.
        output_tool_calls: Vec<ToolCallSnapshot>,
    },
    /// The agent session has finished.
    ///
    /// Carries aggregate usage data (tokens, cost) when the provider reports it.
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
        /// Tool call ID assigned by the LLM (e.g. `toolu_01...` for Anthropic
        /// or the OpenAI tool_call_id). Used to correlate ToolCallStart with
        /// the corresponding LlmRequest's `output_tool_calls`. `None` for
        /// legacy non-native callers.
        tool_call_id: Option<String>,
        /// Full tool call arguments as a JSON string (no truncation as of
        /// Tier 2). Observers that don't want to persist the full payload
        /// can hash or truncate locally.
        arguments: Option<String>,
    },
    /// A tool call has completed with a success/failure outcome.
    ToolCall {
        tool: String,
        tool_call_id: Option<String>,
        duration: Duration,
        success: bool,
        /// Full tool call arguments JSON (duplicated from ToolCallStart so
        /// observers that build a single span from this event get both sides
        /// without needing observer state).
        arguments: Option<String>,
        /// Full tool call result body. `None` when the tool returned no
        /// output (rare) or when the executor's outcome was not yet
        /// available (handler should treat as empty). Already passed
        /// through `scrub_credentials` by the caller.
        result: Option<String>,
    },
    /// The agent produced a final answer for the current user message.
    TurnComplete,
    /// A message was sent or received through a channel.
    ChannelMessage {
        /// Channel name (e.g., `"telegram"`, `"discord"`).
        channel: String,
        /// `"inbound"` or `"outbound"`.
        direction: String,
    },
    /// Periodic heartbeat tick from the runtime keep-alive loop.
    HeartbeatTick,
    /// Response cache hit — an LLM call was avoided.
    CacheHit {
        /// `"hot"` (in-memory) or `"warm"` (SQLite).
        cache_type: String,
        /// Estimated tokens saved by this cache hit.
        tokens_saved: u64,
    },
    /// Response cache miss — the prompt was not found in cache.
    CacheMiss {
        /// `"response"` cache layer that was checked.
        cache_type: String,
    },
    /// An error occurred in a named component.
    Error {
        /// Subsystem where the error originated (e.g., `"provider"`, `"gateway"`).
        component: String,
        /// Human-readable error description. Must not contain secrets or tokens.
        message: String,
    },
    /// A hand has started execution.
    HandStarted { hand_name: String },
    /// A hand has completed execution successfully.
    HandCompleted {
        hand_name: String,
        duration_ms: u64,
        findings_count: usize,
    },
    /// A hand has failed during execution.
    HandFailed {
        hand_name: String,
        error: String,
        duration_ms: u64,
    },
    /// A deployment has started.
    DeploymentStarted {
        /// Identifier for the deployment (e.g., commit SHA or release tag).
        deploy_id: String,
    },
    /// A deployment has completed successfully.
    DeploymentCompleted {
        deploy_id: String,
        /// Commit SHA that was deployed.
        commit_sha: String,
    },
    /// A deployment has failed.
    DeploymentFailed {
        deploy_id: String,
        /// Human-readable failure reason.
        reason: String,
    },
    /// Recovery from a failed deployment has completed.
    RecoveryCompleted { deploy_id: String },
}

// ── Snapshot conversions ──────────────────────────────────────────────
//
// These `From` impls let agent loop call sites build owned snapshots from
// borrowed provider types in a single `.iter().map(Into::into).collect()`
// expression. The conversions are intentionally located in this crate (not
// in `providers/`) so the observability layer is the *consumer* and the
// providers crate has no dependency on `observability`.

impl From<&crate::providers::traits::ChatMessage> for MessageSnapshot {
    fn from(msg: &crate::providers::traits::ChatMessage) -> Self {
        Self {
            role: msg.role.clone(),
            content: msg.content.clone(),
        }
    }
}

impl From<&crate::tools::ToolSpec> for ToolSpecSnapshot {
    fn from(spec: &crate::tools::ToolSpec) -> Self {
        Self {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters_json: serde_json::to_string(&spec.parameters)
                .unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

impl From<&crate::providers::traits::ToolCall> for ToolCallSnapshot {
    fn from(call: &crate::providers::traits::ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments_json: call.arguments.clone(),
        }
    }
}

/// Numeric metrics emitted by the agent runtime.
///
/// Observers can aggregate these into dashboards, alerts, or structured logs.
/// Each variant carries a single scalar value with implicit units.
#[derive(Debug, Clone)]
pub enum ObserverMetric {
    /// Time elapsed for a single LLM or tool request.
    RequestLatency(Duration),
    /// Number of tokens consumed by an LLM call.
    TokensUsed(u64),
    /// Current number of active concurrent sessions.
    ActiveSessions(u64),
    /// Current depth of the inbound message queue.
    QueueDepth(u64),
    /// Duration of a single hand run.
    HandRunDuration {
        hand_name: String,
        duration: Duration,
    },
    /// Number of findings produced by a hand run.
    HandFindingsCount { hand_name: String, count: u64 },
    /// Records a hand run outcome for success-rate tracking.
    HandSuccessRate { hand_name: String, success: bool },
    /// Time elapsed from commit to deployment (lead time for changes).
    DeploymentLeadTime(Duration),
    /// Time elapsed to recover from a failed deployment.
    RecoveryTime(Duration),
}

/// Core observability trait for recording agent runtime telemetry.
///
/// Implement this trait to integrate with any monitoring backend (structured
/// logging, Prometheus, OpenTelemetry, etc.). The agent runtime holds one or
/// more `Observer` instances and calls [`record_event`](Observer::record_event)
/// and [`record_metric`](Observer::record_metric) at key lifecycle points.
///
/// Implementations must be `Send + Sync + 'static` because the observer is
/// shared across async tasks via `Arc`.
pub trait Observer: Send + Sync + 'static {
    /// Record a discrete lifecycle event.
    ///
    /// Called synchronously on the hot path; implementations should avoid
    /// blocking I/O. Buffer events internally and flush asynchronously
    /// when possible.
    fn record_event(&self, event: &ObserverEvent);

    /// Record a numeric metric sample.
    ///
    /// Called synchronously; same non-blocking guidance as
    /// [`record_event`](Observer::record_event).
    fn record_metric(&self, metric: &ObserverMetric);

    /// Flush any buffered telemetry data to the backend.
    ///
    /// The runtime calls this during graceful shutdown. The default
    /// implementation is a no-op, which is appropriate for backends
    /// that write synchronously.
    fn flush(&self) {}

    /// Return the human-readable name of this observer backend.
    ///
    /// Used in logs and diagnostics (e.g., `"console"`, `"prometheus"`,
    /// `"opentelemetry"`).
    fn name(&self) -> &str;

    /// Downcast to `Any` for backend-specific operations.
    ///
    /// Enables callers to access concrete observer types when needed
    /// (e.g., retrieving a Prometheus registry handle for custom metrics).
    fn as_any(&self) -> &dyn std::any::Any;
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

    #[test]
    fn hand_events_recordable() {
        let observer = DummyObserver::default();

        observer.record_event(&ObserverEvent::HandStarted {
            hand_name: "review".into(),
        });
        observer.record_event(&ObserverEvent::HandCompleted {
            hand_name: "review".into(),
            duration_ms: 1500,
            findings_count: 3,
        });
        observer.record_event(&ObserverEvent::HandFailed {
            hand_name: "review".into(),
            error: "timeout".into(),
            duration_ms: 5000,
        });

        assert_eq!(*observer.events.lock(), 3);
    }

    #[test]
    fn hand_metrics_recordable() {
        let observer = DummyObserver::default();

        observer.record_metric(&ObserverMetric::HandRunDuration {
            hand_name: "review".into(),
            duration: Duration::from_millis(1500),
        });
        observer.record_metric(&ObserverMetric::HandFindingsCount {
            hand_name: "review".into(),
            count: 3,
        });
        observer.record_metric(&ObserverMetric::HandSuccessRate {
            hand_name: "review".into(),
            success: true,
        });

        assert_eq!(*observer.metrics.lock(), 3);
    }

    #[test]
    fn hand_event_and_metric_are_cloneable() {
        let event = ObserverEvent::HandCompleted {
            hand_name: "review".into(),
            duration_ms: 500,
            findings_count: 2,
        };
        let metric = ObserverMetric::HandRunDuration {
            hand_name: "review".into(),
            duration: Duration::from_millis(500),
        };

        let cloned_event = event.clone();
        let cloned_metric = metric.clone();

        assert!(matches!(cloned_event, ObserverEvent::HandCompleted { .. }));
        assert!(matches!(
            cloned_metric,
            ObserverMetric::HandRunDuration { .. }
        ));
    }
}
