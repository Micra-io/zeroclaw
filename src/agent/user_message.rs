use chrono::Local;

/// Format a wall-clock timestamp in the canonical user-preamble form.
pub fn format_now() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string()
}

/// Build the enriched user-message content sent to the LLM.
///
/// The per-turn timestamp prefix (`[{now}]`) lives here — outside the
/// system message — so that implicit-caching providers (Qwen, DeepSeek,
/// Groq, OpenAI, Moonshot) keep a byte-identical system prefix across
/// turns and actually reuse the prompt cache.
///
/// `mem_context` is any pre-computed memory/RAG context that the caller
/// wants to prepend (used by the CLI/REST `agent::run` path). The
/// channel daemon path leaves it empty because memory context lives in
/// the system dynamic block there.
pub fn enrich_user_message(now: &str, raw: &str, mem_context: &str) -> String {
    if mem_context.is_empty() {
        format!("[{now}] {raw}")
    } else {
        format!("{mem_context}[{now}] {raw}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enrich_without_context() {
        let out = enrich_user_message("2026-04-17 10:00:00 CEST", "hello", "");
        assert_eq!(out, "[2026-04-17 10:00:00 CEST] hello");
    }

    #[test]
    fn enrich_with_context() {
        let out = enrich_user_message("2026-04-17 10:00:00 CEST", "hello", "## Memory\nfoo\n\n");
        assert_eq!(out, "## Memory\nfoo\n\n[2026-04-17 10:00:00 CEST] hello");
    }

    #[test]
    fn format_now_matches_shape() {
        let s = format_now();
        // Shape: `YYYY-MM-DD HH:MM:SS <tz>`
        assert!(s.starts_with('2'), "year should start with 2, got: {s}");
        assert!(s.len() >= 19, "timestamp too short: {s:?}");
    }
}
