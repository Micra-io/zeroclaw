use crate::multimodal;
use crate::providers::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    Provider, ProviderCapabilities, TokenUsage, ToolCall as ProviderToolCall,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub struct OpenRouterProvider {
    credential: Option<String>,
    timeout_secs: u64,
    max_tokens: Option<u32>,
}

const DEFAULT_OPENROUTER_TIMEOUT_SECS: u64 = 120;
const OPENROUTER_CONNECT_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: MessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}

#[derive(Debug, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

impl CacheControl {
    fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral",
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessagePart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ImageUrl {
        image_url: ImageUrlPart,
    },
}

#[derive(Debug, Serialize)]
struct ImageUrlPart {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ApiChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Serialize)]
struct NativeChatRequest {
    model: String,
    messages: Vec<NativeMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<NativeToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct NativeMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<NativeToolCall>>,
    /// Raw reasoning content from thinking models; pass-through for providers
    /// that require it in assistant tool-call history messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
struct NativeToolSpec {
    #[serde(rename = "type")]
    kind: String,
    function: NativeToolFunctionSpec,
}

#[derive(Debug, Serialize)]
struct NativeToolFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: NativeFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct NativeChatResponse {
    choices: Vec<NativeChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct NativeChoice {
    message: NativeResponseMessage,
}

#[derive(Debug, Deserialize)]
struct NativeResponseMessage {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking models may return output in `reasoning_content`.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<NativeToolCall>>,
}

impl OpenRouterProvider {
    pub fn new(credential: Option<&str>, timeout_secs: Option<u64>) -> Self {
        Self {
            credential: credential.map(ToString::to_string),
            timeout_secs: timeout_secs
                .filter(|secs| *secs > 0)
                .unwrap_or(DEFAULT_OPENROUTER_TIMEOUT_SECS),
            max_tokens: None,
        }
    }

    /// Override the HTTP request timeout for LLM API calls.
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the maximum output tokens for API requests.
    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn convert_tools(tools: Option<&[ToolSpec]>) -> Option<Vec<NativeToolSpec>> {
        let items = tools?;
        if items.is_empty() {
            return None;
        }
        let valid: Vec<NativeToolSpec> = items
            .iter()
            .filter(|tool| is_valid_openai_tool_name(&tool.name))
            .map(|tool| NativeToolSpec {
                kind: "function".to_string(),
                function: NativeToolFunctionSpec {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                },
            })
            .collect();
        if valid.is_empty() { None } else { Some(valid) }
    }

    fn convert_messages(messages: &[ChatMessage]) -> Vec<NativeMessage> {
        messages
            .iter()
            .map(|m| {
                if m.role == "assistant" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content) {
                        if let Some(tool_calls_value) = value.get("tool_calls") {
                            if let Ok(parsed_calls) =
                                serde_json::from_value::<Vec<ProviderToolCall>>(
                                    tool_calls_value.clone(),
                                )
                            {
                                let tool_calls = parsed_calls
                                    .into_iter()
                                    .map(|tc| NativeToolCall {
                                        id: Some(tc.id),
                                        kind: Some("function".to_string()),
                                        function: NativeFunctionCall {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect::<Vec<_>>();
                                let content = value
                                    .get("content")
                                    .and_then(serde_json::Value::as_str)
                                    .map(|value| MessageContent::Text(value.to_string()));
                                let reasoning_content = value
                                    .get("reasoning_content")
                                    .and_then(serde_json::Value::as_str)
                                    .map(ToString::to_string);
                                return NativeMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_call_id: None,
                                    tool_calls: Some(tool_calls),
                                    reasoning_content,
                                };
                            }
                        }
                    }
                }

                if m.role == "tool" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content) {
                        let tool_call_id = value
                            .get("tool_call_id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string);
                        let content = value
                            .get("content")
                            .and_then(serde_json::Value::as_str)
                            .map(|value| MessageContent::Text(value.to_string()))
                            .or_else(|| Some(MessageContent::Text(m.content.clone())));
                        return NativeMessage {
                            role: "tool".to_string(),
                            content,
                            tool_call_id,
                            tool_calls: None,
                            reasoning_content: None,
                        };
                    }
                }

                if m.role == "system" {
                    return NativeMessage {
                        role: "system".to_string(),
                        content: Some(Self::system_message_content(
                            m.stable_prefix.as_deref(),
                            &m.content,
                        )),
                        tool_call_id: None,
                        tool_calls: None,
                        reasoning_content: None,
                    };
                }

                NativeMessage {
                    role: m.role.clone(),
                    content: Some(Self::to_message_content(&m.role, &m.content)),
                    tool_call_id: None,
                    tool_calls: None,
                    reasoning_content: None,
                }
            })
            .collect()
    }

    /// Map a `ChatMessage` to the legacy `Message` used by `chat_with_history`.
    /// System role is routed through `system_message_content` so
    /// `stable_prefix` is honored on this path too — matching the block-format
    /// shape produced by `convert_messages` for the newer native entry points.
    fn to_history_message(m: &ChatMessage) -> Message {
        let content = if m.role == "system" {
            Self::system_message_content(m.stable_prefix.as_deref(), &m.content)
        } else {
            Self::to_message_content(&m.role, &m.content)
        };
        Message {
            role: m.role.clone(),
            content,
        }
    }

    /// Build a system-role `MessageContent` with prompt-caching breakpoint
    /// placed between `stable_prefix` and `content`. When `stable_prefix` is
    /// `Some(non-empty)`, emits two `MessagePart::Text` blocks: stable with
    /// `cache_control: ephemeral`, dynamic without. Otherwise emits a single
    /// cached block. Mirrors the Anthropic adapter's two-`SystemBlock` layout.
    fn system_message_content(stable_prefix: Option<&str>, content: &str) -> MessageContent {
        match stable_prefix {
            Some(stable) if !stable.trim().is_empty() => {
                let mut blocks = Vec::with_capacity(2);
                blocks.push(MessagePart::Text {
                    text: stable.to_string(),
                    cache_control: Some(CacheControl::ephemeral()),
                });
                // Mirror anthropic.rs:569 — skip the dynamic block when content
                // is empty so we never emit a zero-length `{text: ""}` part,
                // which some upstream OpenRouter backends reject.
                if !content.is_empty() {
                    blocks.push(MessagePart::Text {
                        text: content.to_string(),
                        cache_control: None,
                    });
                }
                MessageContent::Parts(blocks)
            }
            _ => MessageContent::Parts(vec![MessagePart::Text {
                text: content.to_string(),
                cache_control: Some(CacheControl::ephemeral()),
            }]),
        }
    }

    fn to_message_content(role: &str, content: &str) -> MessageContent {
        // System role must route through `system_message_content` (via
        // `convert_messages` or `to_history_message`) to keep the
        // `cache_control` breakpoint. Debug-assert to catch any future caller
        // that bypasses the dispatch and silently reintroduces the STORY-008
        // production bug.
        debug_assert_ne!(
            role, "system",
            "system role must be dispatched via system_message_content"
        );
        if role != "user" {
            return MessageContent::Text(content.to_string());
        }

        let (cleaned_text, image_refs) = multimodal::parse_image_markers(content);
        if image_refs.is_empty() {
            return MessageContent::Text(content.to_string());
        }

        let mut parts = Vec::with_capacity(image_refs.len() + 1);
        let trimmed_text = cleaned_text.trim();
        if !trimmed_text.is_empty() {
            parts.push(MessagePart::Text {
                text: trimmed_text.to_string(),
                cache_control: None,
            });
        }

        for image_ref in image_refs {
            parts.push(MessagePart::ImageUrl {
                image_url: ImageUrlPart { url: image_ref },
            });
        }

        MessageContent::Parts(parts)
    }

    fn parse_native_response(message: NativeResponseMessage) -> ProviderChatResponse {
        let reasoning_content = message.reasoning_content.clone();
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ProviderToolCall {
                id: tc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect::<Vec<_>>();

        ProviderChatResponse {
            text: message.content,
            tool_calls,
            usage: None,
            reasoning_content,
        }
    }

    fn compact_sanitized_body_snippet(body: &str) -> String {
        super::sanitize_api_error(body)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    async fn read_response_body(
        provider_name: &str,
        response: reqwest::Response,
    ) -> anyhow::Result<String> {
        response.text().await.map_err(|error| {
            let sanitized = super::sanitize_api_error(&error.to_string());
            anyhow::anyhow!(
                "{provider_name} transport error while reading response body: {sanitized}"
            )
        })
    }

    fn parse_response_body<T: DeserializeOwned>(
        provider_name: &str,
        body: &str,
        kind: &str,
    ) -> anyhow::Result<T> {
        serde_json::from_str::<T>(body).map_err(|error| {
            let snippet = Self::compact_sanitized_body_snippet(body);
            anyhow::anyhow!(
                "{provider_name} API returned an unexpected {kind} payload: {error}; body={snippet}"
            )
        })
    }

    fn http_client(&self) -> Client {
        crate::config::build_runtime_proxy_client_with_timeouts(
            "provider.openrouter",
            self.timeout_secs,
            OPENROUTER_CONNECT_TIMEOUT_SECS,
        )
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: true,
            prompt_caching: false,
        }
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        // Hit a lightweight endpoint to establish TLS + HTTP/2 connection pool.
        // This prevents the first real chat request from timing out on cold start.
        if let Some(credential) = self.credential.as_ref() {
            self.http_client()
                .get("https://openrouter.ai/api/v1/auth/key")
                .header("Authorization", format!("Bearer {credential}"))
                .send()
                .await?
                .error_for_status()?;
        }
        Ok(())
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let credential = self.credential.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenRouter API key not set. Run `zeroclaw onboard` or set OPENROUTER_API_KEY env var."))?;

        let mut messages = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(Message {
                role: "system".to_string(),
                content: MessageContent::Text(sys.to_string()),
            });
        }

        messages.push(Message {
            role: "user".to_string(),
            content: Self::to_message_content("user", message),
        });

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature,
            max_tokens: self.max_tokens,
        };

        let response = self
            .http_client()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {credential}"))
            .header("HTTP-Referer", "https://github.com/zeroclaw-labs/zeroclaw")
            .header("X-Title", "ZeroClaw")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let body = Self::read_response_body("OpenRouter", response).await?;
        let chat_response =
            Self::parse_response_body::<ApiChatResponse>("OpenRouter", &body, "chat-completions")?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let credential = self.credential.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenRouter API key not set. Run `zeroclaw onboard` or set OPENROUTER_API_KEY env var."))?;

        let api_messages: Vec<Message> = messages.iter().map(Self::to_history_message).collect();

        let request = ChatRequest {
            model: model.to_string(),
            messages: api_messages,
            temperature,
            max_tokens: self.max_tokens,
        };

        let response = self
            .http_client()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {credential}"))
            .header("HTTP-Referer", "https://github.com/zeroclaw-labs/zeroclaw")
            .header("X-Title", "ZeroClaw")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let body = Self::read_response_body("OpenRouter", response).await?;
        let chat_response =
            Self::parse_response_body::<ApiChatResponse>("OpenRouter", &body, "chat-completions")?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
            "OpenRouter API key not set. Run `zeroclaw onboard` or set OPENROUTER_API_KEY env var."
        )
        })?;

        let tools = Self::convert_tools(request.tools);
        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(request.messages),
            temperature,
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            tools,
            max_tokens: self.max_tokens,
        };

        let response = self
            .http_client()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {credential}"))
            .header("HTTP-Referer", "https://github.com/zeroclaw-labs/zeroclaw")
            .header("X-Title", "ZeroClaw")
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let body = Self::read_response_body("OpenRouter", response).await?;
        let native_response =
            Self::parse_response_body::<NativeChatResponse>("OpenRouter", &body, "native chat")?;
        // OpenRouter cached_tokens = read-only (no creation billing).
        // cached_input_tokens = read + creation = read + 0.
        let usage = native_response.usage.map(|u| {
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_read_input_tokens: cached,
                cache_creation_input_tokens: None,
            }
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "OpenRouter API key not set. Run `zeroclaw onboard` or set OPENROUTER_API_KEY env var."
            )
        })?;

        // Convert tool JSON values to NativeToolSpec
        let native_tools: Option<Vec<NativeToolSpec>> = if tools.is_empty() {
            None
        } else {
            let specs: Vec<NativeToolSpec> = tools
                .iter()
                .filter_map(|t| {
                    let func = t.get("function")?;
                    Some(NativeToolSpec {
                        kind: "function".to_string(),
                        function: NativeToolFunctionSpec {
                            name: func.get("name")?.as_str()?.to_string(),
                            description: func
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string(),
                            parameters: func
                                .get("parameters")
                                .cloned()
                                .unwrap_or(serde_json::json!({})),
                        },
                    })
                })
                .collect();
            if specs.is_empty() { None } else { Some(specs) }
        };

        // Convert ChatMessage to NativeMessage, preserving structured assistant/tool entries
        // when history contains native tool-call metadata.
        let native_messages = Self::convert_messages(messages);

        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: native_messages,
            temperature,
            tool_choice: native_tools.as_ref().map(|_| "auto".to_string()),
            tools: native_tools,
            max_tokens: self.max_tokens,
        };

        let response = self
            .http_client()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {credential}"))
            .header("HTTP-Referer", "https://github.com/zeroclaw-labs/zeroclaw")
            .header("X-Title", "ZeroClaw")
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let body = Self::read_response_body("OpenRouter", response).await?;
        let native_response =
            Self::parse_response_body::<NativeChatResponse>("OpenRouter", &body, "native chat")?;
        // OpenRouter cached_tokens = read-only (no creation billing).
        // cached_input_tokens = read + creation = read + 0.
        let usage = native_response.usage.map(|u| {
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_read_input_tokens: cached,
                cache_creation_input_tokens: None,
            }
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }
}

/// Check if a tool name is valid for OpenAI-compatible APIs.
/// Must match `^[a-zA-Z0-9_-]{1,64}$`.
fn is_valid_openai_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::traits::{ChatMessage, Provider};

    #[test]
    fn capabilities_report_vision_support() {
        let provider = OpenRouterProvider::new(Some("openrouter-test-credential"), None);
        let caps = <OpenRouterProvider as Provider>::capabilities(&provider);
        assert!(caps.native_tool_calling);
        assert!(caps.vision);
    }

    #[test]
    fn creates_with_key() {
        let provider = OpenRouterProvider::new(Some("openrouter-test-credential"), None);
        assert_eq!(
            provider.credential.as_deref(),
            Some("openrouter-test-credential")
        );
    }

    #[test]
    fn creates_without_key() {
        let provider = OpenRouterProvider::new(None, None);
        assert!(provider.credential.is_none());
    }

    #[test]
    fn uses_configured_timeout_when_provided() {
        let provider = OpenRouterProvider::new(Some("openrouter-test-credential"), Some(1200));
        assert_eq!(provider.timeout_secs, 1200);
    }

    #[test]
    fn falls_back_to_default_timeout_for_zero() {
        let provider = OpenRouterProvider::new(Some("openrouter-test-credential"), Some(0));
        assert_eq!(provider.timeout_secs, DEFAULT_OPENROUTER_TIMEOUT_SECS);
    }

    #[tokio::test]
    async fn warmup_without_key_is_noop() {
        let provider = OpenRouterProvider::new(None, None);
        let result = provider.warmup().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_key() {
        let provider = OpenRouterProvider::new(None, None);
        let result = provider
            .chat_with_system(Some("system"), "hello", "openai/gpt-4o", 0.2)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[tokio::test]
    async fn chat_with_history_fails_without_key() {
        let provider = OpenRouterProvider::new(None, None);
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "be concise".into(),
                stable_prefix: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "hello".into(),
                stable_prefix: None,
            },
        ];

        let result = provider
            .chat_with_history(&messages, "anthropic/claude-sonnet-4", 0.7)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[test]
    fn chat_request_serializes_with_system_and_user() {
        let request = ChatRequest {
            model: "anthropic/claude-sonnet-4".into(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: MessageContent::Text("You are helpful".into()),
                },
                Message {
                    role: "user".into(),
                    content: MessageContent::Text("Summarize this".into()),
                },
            ],
            temperature: 0.5,
            max_tokens: None,
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("anthropic/claude-sonnet-4"));
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn chat_request_serializes_history_messages() {
        let messages = [
            ChatMessage {
                role: "assistant".into(),
                content: "Previous answer".into(),
                stable_prefix: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "Follow-up".into(),
                stable_prefix: None,
            },
        ];

        let request = ChatRequest {
            model: "google/gemini-2.5-pro".into(),
            messages: messages
                .iter()
                .map(|msg| Message {
                    role: msg.role.clone(),
                    content: MessageContent::Text(msg.content.clone()),
                })
                .collect(),
            temperature: 0.0,
            max_tokens: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("google/gemini-2.5-pro"));
    }

    #[test]
    fn response_deserializes_single_choice() {
        let json = r#"{"choices":[{"message":{"content":"Hi from OpenRouter"}}]}"#;

        let response: ApiChatResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, "Hi from OpenRouter");
    }

    #[test]
    fn response_deserializes_empty_choices() {
        let json = r#"{"choices":[]}"#;

        let response: ApiChatResponse = serde_json::from_str(json).unwrap();

        assert!(response.choices.is_empty());
    }

    #[test]
    fn parse_chat_response_body_reports_sanitized_snippet() {
        let body = r#"{"choices":"invalid","api_key":"sk-test-secret-value"}"#;
        let err = OpenRouterProvider::parse_response_body::<ApiChatResponse>(
            "OpenRouter",
            body,
            "chat-completions",
        )
        .expect_err("payload should fail");
        let msg = err.to_string();

        assert!(msg.contains("OpenRouter API returned an unexpected chat-completions payload"));
        assert!(msg.contains("body="));
        assert!(msg.contains("[REDACTED]"));
        assert!(!msg.contains("sk-test-secret-value"));
    }

    #[test]
    fn parse_native_response_body_reports_sanitized_snippet() {
        let body = r#"{"choices":123,"api_key":"sk-another-secret"}"#;
        let err = OpenRouterProvider::parse_response_body::<NativeChatResponse>(
            "OpenRouter",
            body,
            "native chat",
        )
        .expect_err("payload should fail");
        let msg = err.to_string();

        assert!(msg.contains("OpenRouter API returned an unexpected native chat payload"));
        assert!(msg.contains("body="));
        assert!(msg.contains("[REDACTED]"));
        assert!(!msg.contains("sk-another-secret"));
    }

    #[tokio::test]
    async fn chat_with_tools_fails_without_key() {
        let provider = OpenRouterProvider::new(None, None);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "What is the date?".into(),
            stable_prefix: None,
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
            }
        })];

        let result = provider
            .chat_with_tools(&messages, &tools, "deepseek/deepseek-chat", 0.5)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[test]
    fn native_response_deserializes_with_tool_calls() {
        let json = r#"{
            "choices":[{
                "message":{
                    "content":null,
                    "tool_calls":[
                        {"id":"call_123","type":"function","function":{"name":"get_price","arguments":"{\"symbol\":\"BTC\"}"}}
                    ]
                }
            }]
        }"#;

        let response: NativeChatResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.choices.len(), 1);
        let message = &response.choices[0].message;
        assert!(message.content.is_none());
        let tool_calls = message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_123"));
        assert_eq!(tool_calls[0].function.name, "get_price");
        assert_eq!(tool_calls[0].function.arguments, "{\"symbol\":\"BTC\"}");
    }

    #[test]
    fn native_response_deserializes_with_text_and_tool_calls() {
        let json = r#"{
            "choices":[{
                "message":{
                    "content":"I'll get that for you.",
                    "tool_calls":[
                        {"id":"call_456","type":"function","function":{"name":"shell","arguments":"{\"command\":\"date\"}"}}
                    ]
                }
            }]
        }"#;

        let response: NativeChatResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.choices.len(), 1);
        let message = &response.choices[0].message;
        assert_eq!(message.content.as_deref(), Some("I'll get that for you."));
        let tool_calls = message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "shell");
    }

    #[test]
    fn parse_native_response_converts_to_chat_response() {
        let message = NativeResponseMessage {
            content: Some("Here you go.".into()),
            reasoning_content: None,
            tool_calls: Some(vec![NativeToolCall {
                id: Some("call_789".into()),
                kind: Some("function".into()),
                function: NativeFunctionCall {
                    name: "file_read".into(),
                    arguments: r#"{"path":"test.txt"}"#.into(),
                },
            }]),
        };

        let response = OpenRouterProvider::parse_native_response(message);

        assert_eq!(response.text.as_deref(), Some("Here you go."));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_789");
        assert_eq!(response.tool_calls[0].name, "file_read");
    }

    #[test]
    fn convert_messages_parses_assistant_tool_call_payload() {
        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: r#"{"content":"Using tool","tool_calls":[{"id":"call_abc","name":"shell","arguments":"{\"command\":\"pwd\"}"}]}"#
                .into(),
            stable_prefix: None,
        }];

        let converted = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "assistant");
        assert_eq!(
            converted[0]
                .content
                .as_ref()
                .and_then(|content| match content {
                    MessageContent::Text(value) => Some(value.as_str()),
                    MessageContent::Parts(_) => None,
                }),
            Some("Using tool")
        );

        let tool_calls = converted[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(tool_calls[0].function.name, "shell");
        assert_eq!(tool_calls[0].function.arguments, r#"{"command":"pwd"}"#);
    }

    #[test]
    fn convert_messages_parses_tool_result_payload() {
        let messages = vec![ChatMessage {
            role: "tool".into(),
            content: r#"{"tool_call_id":"call_xyz","content":"done"}"#.into(),
            stable_prefix: None,
        }];

        let converted = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_xyz"));
        assert_eq!(
            converted[0]
                .content
                .as_ref()
                .and_then(|content| match content {
                    MessageContent::Text(value) => Some(value.as_str()),
                    MessageContent::Parts(_) => None,
                }),
            Some("done")
        );
        assert!(converted[0].tool_calls.is_none());
    }

    #[test]
    fn to_message_content_converts_image_markers_to_openai_parts() {
        let content = "Describe this\n\n[IMAGE:data:image/png;base64,abcd]";
        let value =
            serde_json::to_value(OpenRouterProvider::to_message_content("user", content)).unwrap();
        let parts = value
            .as_array()
            .expect("multimodal content should be an array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Describe this");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,abcd");
    }

    #[test]
    fn native_response_parses_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 15}
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(42));
        assert_eq!(usage.completion_tokens, Some(15));
    }

    #[test]
    fn native_response_parses_without_usage() {
        let json = r#"{"choices": [{"message": {"content": "Hello"}}]}"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // prompt caching: response-side deserialization (Unit 1)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn usage_deserializes_cached_tokens() {
        let json = r#"{
            "prompt_tokens": 25000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 20000}
        }"#;
        let usage: UsageInfo = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, Some(25000));
        assert_eq!(usage.completion_tokens, Some(500));
        let details = usage.prompt_tokens_details.unwrap();
        assert_eq!(details.cached_tokens, Some(20000));
    }

    #[test]
    fn usage_deserializes_cached_tokens_zero() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0}
        }"#;
        let usage: UsageInfo = serde_json::from_str(json).unwrap();
        let details = usage.prompt_tokens_details.unwrap();
        assert_eq!(details.cached_tokens, Some(0));
    }

    #[test]
    fn usage_deserializes_without_prompt_tokens_details() {
        let json = r#"{"prompt_tokens": 100, "completion_tokens": 50}"#;
        let usage: UsageInfo = serde_json::from_str(json).unwrap();
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn usage_deserializes_empty_prompt_tokens_details() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {}
        }"#;
        let usage: UsageInfo = serde_json::from_str(json).unwrap();
        let details = usage.prompt_tokens_details.unwrap();
        assert!(details.cached_tokens.is_none());
    }

    #[test]
    fn native_response_deserializes_with_cached_tokens() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 25000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 20000}
            }
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        let details = usage.prompt_tokens_details.unwrap();
        assert_eq!(details.cached_tokens, Some(20000));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // prompt caching: request-side serialization (Unit 2)
    // ═══════════════════════════════════════════════════════════════════════

    // Note: STORY-008's `system_message_serializes_as_content_block_with_cache_control`
    // was deleted — it tested the system branch of `to_message_content` which
    // STORY-009 removed. Equivalent coverage now lives in
    // `system_message_without_stable_prefix_emits_single_cached_block`, which
    // exercises the live `system_message_content` helper.

    #[test]
    fn user_message_serializes_as_plain_string() {
        let content = OpenRouterProvider::to_message_content("user", "Hello");
        let json = serde_json::to_value(&content).unwrap();
        assert!(json.is_string(), "user content should be a plain string");
        assert_eq!(json.as_str().unwrap(), "Hello");
    }

    #[test]
    fn assistant_message_serializes_as_plain_string() {
        let content = OpenRouterProvider::to_message_content("assistant", "Hi there.");
        let json = serde_json::to_value(&content).unwrap();
        assert!(
            json.is_string(),
            "assistant content should be a plain string"
        );
        assert_eq!(json.as_str().unwrap(), "Hi there.");
    }

    #[test]
    fn cache_control_not_serialized_for_user_image_text_part() {
        let content = OpenRouterProvider::to_message_content(
            "user",
            "Describe this\n\n[IMAGE:data:image/png;base64,abcd]",
        );
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().expect("multimodal content should be array");
        let text_part = &parts[0];
        assert_eq!(text_part["type"], "text");
        assert!(
            text_part.get("cache_control").is_none() || text_part["cache_control"].is_null(),
            "cache_control should not appear on user image text parts"
        );
    }

    #[test]
    fn full_native_request_serializes_system_as_blocks_user_as_string() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "Be helpful".into(),
                stable_prefix: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
                stable_prefix: None,
            },
        ];
        let native = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(native.len(), 2);

        // System message should be Parts
        let sys_json = serde_json::to_value(&native[0].content).unwrap();
        let sys_content = sys_json.as_array().expect("system content should be array");
        assert_eq!(sys_content[0]["cache_control"]["type"], "ephemeral");

        // User message should be Text
        let user_json = serde_json::to_value(&native[1].content).unwrap();
        assert!(user_json.is_string(), "user content should be string");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // prompt caching: stable_prefix split (STORY-009)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn system_message_with_stable_prefix_emits_two_blocks() {
        let content = OpenRouterProvider::system_message_content(Some("STABLE"), "DYNAMIC");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().expect("system content should be an array");
        assert_eq!(parts.len(), 2, "expected stable + dynamic blocks");

        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "STABLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");

        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["text"], "DYNAMIC");
    }

    #[test]
    fn system_message_without_stable_prefix_emits_single_cached_block() {
        let content = OpenRouterProvider::system_message_content(None, "WHOLE");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().expect("system content should be an array");
        assert_eq!(parts.len(), 1, "no stable prefix → single block");

        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "WHOLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn system_message_with_stable_prefix_and_empty_content_omits_dynamic_block() {
        // Matches the Anthropic adapter guard at anthropic.rs:569 which skips
        // the dynamic SystemBlock when content is empty. Without this guard,
        // strict OpenAI-compatible upstream models (GPT-4o, DeepSeek-v3) may
        // 400 on a `{text: ""}` content part (correctness COR-001, adversarial
        // ADV-002).
        let content = OpenRouterProvider::system_message_content(Some("STABLE"), "");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().unwrap();
        assert_eq!(parts.len(), 1, "empty content → omit dynamic block");
        assert_eq!(parts[0]["text"], "STABLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn system_message_with_empty_stable_prefix_falls_back_to_single_block() {
        let content = OpenRouterProvider::system_message_content(Some(""), "WHOLE");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().expect("system content should be an array");
        assert_eq!(parts.len(), 1, "empty prefix collapses to single block");

        assert_eq!(parts[0]["text"], "WHOLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn system_message_with_whitespace_only_stable_prefix_falls_back_to_single_block() {
        // A whitespace-only prefix would emit a whitespace-only stable block
        // that defeats prefix caching. `concatenated_content` in traits.rs also
        // treats whitespace-only as empty via `trim_end()` — this keeps the
        // two code paths consistent (testing T-001, correctness COR-002).
        let content = OpenRouterProvider::system_message_content(Some("   \n"), "WHOLE");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().unwrap();
        assert_eq!(
            parts.len(),
            1,
            "whitespace-only prefix collapses to single block"
        );
        assert_eq!(parts[0]["text"], "WHOLE");
    }

    #[test]
    fn system_message_blocks_preserve_stable_then_dynamic_order() {
        // Cache breakpoint must sit AFTER stable bytes — stable block first,
        // dynamic block second. Reversing the order would defeat the purpose
        // because OpenRouter's prefix cache lookup fails on any byte change
        // before the cache_control marker.
        let content =
            OpenRouterProvider::system_message_content(Some("STABLE_FIRST"), "DYNAMIC_SECOND");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().unwrap();

        assert_eq!(parts[0]["text"], "STABLE_FIRST");
        assert!(parts[0].get("cache_control").is_some());
        assert_eq!(parts[1]["text"], "DYNAMIC_SECOND");
    }

    #[test]
    fn convert_messages_system_with_stable_prefix_emits_two_blocks() {
        // End-to-end: ChatMessage with stable_prefix → convert_messages →
        // NativeMessage[0].content is a 2-block Parts array with cache_control
        // only on block 0. This is the test that would have caught the
        // production bug (cache_read_input_tokens always 0).
        let messages = vec![
            ChatMessage::system_with_stable_prefix("STABLE", "DYNAMIC"),
            ChatMessage::user("Hi"),
        ];
        let native = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(native.len(), 2);

        let sys_json = serde_json::to_value(&native[0].content).unwrap();
        let parts = sys_json.as_array().expect("system content should be array");
        assert_eq!(parts.len(), 2, "stable_prefix should produce 2 blocks");
        assert_eq!(parts[0]["text"], "STABLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(parts[1]["text"], "DYNAMIC");
        assert!(parts[1].get("cache_control").is_none());

        // User message unchanged — plain string
        let user_json = serde_json::to_value(&native[1].content).unwrap();
        assert!(user_json.is_string());
    }

    #[test]
    fn convert_messages_applies_stable_prefix_per_system_message() {
        // Defensive: OpenRouter allows multiple system messages and applies
        // stable_prefix per-message, unlike Anthropic which captures only the
        // first. Guards against accidental refactor to "first system only".
        let messages = vec![
            ChatMessage::system("first plain"),
            ChatMessage::system_with_stable_prefix("second-stable", "second-dynamic"),
            ChatMessage::user("Hi"),
        ];
        let native = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(native.len(), 3);

        // First system: 1 block (no stable_prefix)
        let first = serde_json::to_value(&native[0].content).unwrap();
        let first_parts = first.as_array().unwrap();
        assert_eq!(first_parts.len(), 1);
        assert_eq!(first_parts[0]["text"], "first plain");

        // Second system: 2 blocks (stable_prefix set)
        let second = serde_json::to_value(&native[1].content).unwrap();
        let second_parts = second.as_array().unwrap();
        assert_eq!(second_parts.len(), 2);
        assert_eq!(second_parts[0]["text"], "second-stable");
        assert_eq!(second_parts[1]["text"], "second-dynamic");
    }

    #[test]
    fn chat_request_body_contains_split_system_blocks_when_stable_prefix_set() {
        // Integration: serialize a full request body and assert the on-wire
        // shape. This is the test that would have caught the production bug
        // (cache_read_input_tokens always 0) end-to-end.
        let messages = vec![
            ChatMessage::system_with_stable_prefix("STABLE_BYTES", "DYNAMIC_BYTES"),
            ChatMessage::user("Hello"),
        ];
        let native = OpenRouterProvider::convert_messages(&messages);
        let request = NativeChatRequest {
            model: "test-model".to_string(),
            messages: native,
            temperature: 0.0,
            tools: None,
            tool_choice: None,
            max_tokens: None,
        };
        let body = serde_json::to_value(&request).unwrap();
        let messages = body["messages"].as_array().unwrap();

        // First message: system with split content
        assert_eq!(messages[0]["role"], "system");
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(
            parts.len(),
            2,
            "system must serialize as 2 blocks on the wire"
        );
        assert_eq!(parts[0]["text"], "STABLE_BYTES");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(parts[1]["text"], "DYNAMIC_BYTES");
        assert!(parts[1].get("cache_control").is_none());
    }

    #[test]
    fn convert_messages_user_assistant_tool_unchanged_when_stable_prefix_present() {
        // ChatMessage allows stable_prefix on any role structurally. For
        // non-system roles it must be silently ignored — serialization
        // byte-identical to the no-prefix case.
        let with_prefix = vec![
            ChatMessage {
                role: "user".into(),
                content: "U".into(),
                stable_prefix: Some("ignored".into()),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "A".into(),
                stable_prefix: Some("ignored".into()),
            },
        ];
        let without_prefix = vec![ChatMessage::user("U"), ChatMessage::assistant("A")];

        let with = OpenRouterProvider::convert_messages(&with_prefix);
        let without = OpenRouterProvider::convert_messages(&without_prefix);

        assert_eq!(
            serde_json::to_value(&with).unwrap(),
            serde_json::to_value(&without).unwrap(),
            "stable_prefix on non-system roles must not affect serialization"
        );
    }

    #[test]
    fn to_history_message_system_with_stable_prefix_emits_two_blocks() {
        // chat_with_history builds legacy Message structs via to_history_message.
        // Before the fix, system messages on this path went through
        // to_message_content and lost cache_control — a silent regression that
        // reviewers (api-contract ACR-001, adversarial ADV-001) caught.
        let msg = ChatMessage::system_with_stable_prefix("STABLE", "DYNAMIC");
        let api = OpenRouterProvider::to_history_message(&msg);
        assert_eq!(api.role, "system");

        let json = serde_json::to_value(&api.content).unwrap();
        let parts = json.as_array().expect("system must serialize as Parts");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "STABLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(parts[1]["text"], "DYNAMIC");
        assert!(parts[1].get("cache_control").is_none());
    }

    #[test]
    fn to_history_message_system_without_stable_prefix_emits_single_cached_block() {
        let msg = ChatMessage::system("WHOLE");
        let api = OpenRouterProvider::to_history_message(&msg);
        let json = serde_json::to_value(&api.content).unwrap();
        let parts = json.as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "WHOLE");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn to_history_message_user_serializes_as_plain_string() {
        let msg = ChatMessage::user("Hi");
        let api = OpenRouterProvider::to_history_message(&msg);
        let json = serde_json::to_value(&api.content).unwrap();
        assert!(json.is_string());
        assert_eq!(json.as_str().unwrap(), "Hi");
    }

    #[test]
    fn system_message_dynamic_block_omits_cache_control_field() {
        // The dynamic block's cache_control must be absent from the serialized
        // JSON (not just `null`) — relies on `skip_serializing_if` on
        // `MessagePart::Text::cache_control`. If the field were present the
        // OpenRouter API would treat it as a second cache breakpoint, which
        // would invalidate the prefix cache on every dynamic-content change.
        let content = OpenRouterProvider::system_message_content(Some("S"), "D");
        let json = serde_json::to_value(&content).unwrap();
        let parts = json.as_array().unwrap();

        assert!(
            parts[1].get("cache_control").is_none(),
            "dynamic block must NOT carry cache_control: got {parts:?}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STORY-011 Phase C regression guards
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn story_011_system_message_has_no_timestamp_no_model_and_one_cache_control() {
        // STORY-011 invariant: the serialized OpenRouter system message must
        // carry neither the old `## Current Date & Time` section nor the
        // `Model:` label (both now live in the user-message preamble) AND
        // must expose exactly ONE `ephemeral` cache_control breakpoint on
        // the stable block. Implicit-caching providers (Qwen, DeepSeek,
        // Groq, OpenAI, Moonshot) ignore cache_control and key on the
        // longest byte-identical prefix; a timestamp or per-model label in
        // the system message would drop cache hits to 0% on every turn.
        let stable = "## Identity\n\nYou are ZeroClaw.\n\n\
                      ## Tools\n\n- shell: Run commands\n\n\
                      ## Safety\n\n- NEVER repeat credentials.\n\n\
                      ## Channel Capabilities\n\nrunning as a messaging bot\n\n\
                      ## Host Environment\n\nHost: testhost | OS: macos\n";
        let dynamic = "";

        let msg = ChatMessage::system_with_stable_prefix(stable, dynamic);
        let api = OpenRouterProvider::to_history_message(&msg);
        let json = serde_json::to_value(&api.content).unwrap();
        let json_str = serde_json::to_string(&json).unwrap();

        assert!(
            !json_str.contains("## Current Date & Time"),
            "STORY-011: serialized system message must not contain `## Current Date & Time`; got: {json_str}"
        );
        assert!(
            !json_str.contains("Model:"),
            "STORY-011: serialized system message must not contain `Model:` label (moved to user preamble); got: {json_str}"
        );

        let parts = json.as_array().expect("system content must be Parts");
        let ephemeral_count = parts
            .iter()
            .filter(|p| {
                p.get("cache_control").and_then(|c| c.get("type"))
                    == Some(&serde_json::json!("ephemeral"))
            })
            .count();
        assert_eq!(
            ephemeral_count, 1,
            "STORY-011: exactly one MessagePart must carry cache_control.type=ephemeral; got parts: {parts:?}"
        );
    }

    #[test]
    fn story_011_single_ephemeral_breakpoint_covers_stable_block() {
        // Regression guard — verifies the cache breakpoint covers the stable
        // content, not the dynamic. Anthropic/Claude respect cache_control
        // explicitly; Qwen & other implicit-caching providers ignore it, but
        // if a future change migrated the breakpoint to the dynamic block,
        // Anthropic would cache mutating content and miss on every turn.
        let stable = "## Identity\n\nYou are ZeroClaw.\n\n\
                      ## Host Environment\n\nHost: testhost | OS: macos\n";
        let dynamic = "hello";

        let msg = ChatMessage::system_with_stable_prefix(stable, dynamic);
        let api = OpenRouterProvider::to_history_message(&msg);
        let json = serde_json::to_value(&api.content).unwrap();
        let parts = json.as_array().expect("system content must be Parts");

        let cached: Vec<_> = parts
            .iter()
            .filter(|p| {
                p.get("cache_control").and_then(|c| c.get("type"))
                    == Some(&serde_json::json!("ephemeral"))
            })
            .collect();
        assert_eq!(
            cached.len(),
            1,
            "expected exactly one ephemeral breakpoint, got {cached:?}"
        );
        let cached_text = cached[0]["text"]
            .as_str()
            .expect("cached part must have text");
        assert!(
            cached_text.contains("## Host Environment"),
            "the single ephemeral breakpoint must cover the stable block (which contains `## Host Environment`); got cached text: {cached_text}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // prompt caching: response-side token mapping (Unit 3)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn native_response_maps_cached_tokens_to_token_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 25000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 15000}
            }
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.map(|u| {
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_read_input_tokens: cached,
                cache_creation_input_tokens: None,
            }
        });
        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, Some(25000));
        assert_eq!(usage.output_tokens, Some(500));
        assert_eq!(usage.cached_input_tokens, Some(15000));
        assert_eq!(usage.cache_read_input_tokens, Some(15000));
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn native_response_maps_none_when_no_prompt_tokens_details() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.map(|u| {
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_read_input_tokens: cached,
                cache_creation_input_tokens: None,
            }
        });
        let usage = usage.unwrap();
        assert!(usage.cached_input_tokens.is_none());
        assert!(usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn native_response_maps_zero_cached_tokens_as_some_zero() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.map(|u| {
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_input_tokens: cached,
                cache_read_input_tokens: cached,
                cache_creation_input_tokens: None,
            }
        });
        let usage = usage.unwrap();
        assert_eq!(usage.cached_input_tokens, Some(0));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // reasoning_content pass-through tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_native_response_captures_reasoning_content() {
        let message = NativeResponseMessage {
            content: Some("answer".into()),
            reasoning_content: Some("thinking step".into()),
            tool_calls: Some(vec![NativeToolCall {
                id: Some("call_1".into()),
                kind: Some("function".into()),
                function: NativeFunctionCall {
                    name: "shell".into(),
                    arguments: "{}".into(),
                },
            }]),
        };
        let parsed = OpenRouterProvider::parse_native_response(message);
        assert_eq!(parsed.reasoning_content.as_deref(), Some("thinking step"));
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[test]
    fn parse_native_response_none_reasoning_content_for_normal_model() {
        let message = NativeResponseMessage {
            content: Some("hello".into()),
            reasoning_content: None,
            tool_calls: None,
        };
        let parsed = OpenRouterProvider::parse_native_response(message);
        assert!(parsed.reasoning_content.is_none());
    }

    #[test]
    fn native_response_deserializes_reasoning_content() {
        let json = r#"{
            "choices":[{
                "message":{
                    "content":"answer",
                    "reasoning_content":"deep thought",
                    "tool_calls":[
                        {"id":"call_r1","type":"function","function":{"name":"shell","arguments":"{}"}}
                    ]
                }
            }]
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let message = &resp.choices[0].message;
        assert_eq!(message.reasoning_content.as_deref(), Some("deep thought"));
    }

    #[test]
    fn convert_messages_round_trips_reasoning_content() {
        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{}"
            }],
            "reasoning_content": "Let me think..."
        });

        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: history_json.to_string(),
            stable_prefix: None,
        }];
        let native = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(native.len(), 1);
        assert_eq!(
            native[0].reasoning_content.as_deref(),
            Some("Let me think...")
        );
    }

    #[test]
    fn convert_messages_no_reasoning_content_when_absent() {
        let history_json = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{
                "id": "tc_1",
                "name": "shell",
                "arguments": "{}"
            }]
        });

        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: history_json.to_string(),
            stable_prefix: None,
        }];
        let native = OpenRouterProvider::convert_messages(&messages);
        assert_eq!(native.len(), 1);
        assert!(native[0].reasoning_content.is_none());
    }

    #[test]
    fn native_message_omits_reasoning_content_when_none() {
        let msg = NativeMessage {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text("hi".into())),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("reasoning_content"));
    }

    #[test]
    fn native_message_includes_reasoning_content_when_some() {
        let msg = NativeMessage {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text("hi".into())),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: Some("thinking...".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("reasoning_content"));
        assert!(json.contains("thinking..."));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // timeout_secs configuration tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn default_timeout_is_120() {
        let provider = OpenRouterProvider::new(Some("key"), None);
        assert_eq!(provider.timeout_secs, 120);
    }

    #[test]
    fn with_timeout_secs_overrides_default() {
        let provider = OpenRouterProvider::new(Some("key"), None).with_timeout_secs(300);
        assert_eq!(provider.timeout_secs, 300);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // tool name validation tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn valid_openai_tool_names() {
        assert!(is_valid_openai_tool_name("shell"));
        assert!(is_valid_openai_tool_name("file_read"));
        assert!(is_valid_openai_tool_name("web-search"));
        assert!(is_valid_openai_tool_name("Tool123"));
        assert!(is_valid_openai_tool_name("a"));
    }

    #[test]
    fn invalid_openai_tool_names() {
        assert!(!is_valid_openai_tool_name(""));
        assert!(!is_valid_openai_tool_name("mcp:server.tool"));
        assert!(!is_valid_openai_tool_name("node.js"));
        assert!(!is_valid_openai_tool_name("tool name"));
        assert!(!is_valid_openai_tool_name(
            "this_tool_name_is_way_too_long_and_exceeds_the_sixty_four_character_limit_xxxxx"
        ));
    }

    #[test]
    fn convert_tools_skips_invalid_names() {
        use crate::tools::ToolSpec;

        let tools = vec![
            ToolSpec {
                name: "valid_tool".into(),
                description: "A valid tool".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolSpec {
                name: "mcp:server.bad".into(),
                description: "Invalid name".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolSpec {
                name: "another-valid".into(),
                description: "Also valid".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];

        let result = OpenRouterProvider::convert_tools(Some(&tools)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "valid_tool");
        assert_eq!(result[1].function.name, "another-valid");
    }

    #[test]
    fn convert_tools_returns_none_when_all_invalid() {
        use crate::tools::ToolSpec;

        let tools = vec![ToolSpec {
            name: "mcp:bad.name".into(),
            description: "Invalid".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        assert!(OpenRouterProvider::convert_tools(Some(&tools)).is_none());
    }
}
