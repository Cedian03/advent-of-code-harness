use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result};
use llm_http::{ChatRequest, ToolCall, ToolDefinition, Usage};
use serde::Serialize;
use tracing::{debug, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }

    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// One model turn: text, any tool calls, and the stop reason.
#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Usage,
}

pub trait Model: Send + Sync {
    fn name(&self) -> &str;

    /// Ask the model for the next step, streaming text through `on_delta`.
    fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Completion>;
}

/// Chat Completions adapter over [`llm_http`].
///
/// Works with OpenAI and any OpenAI-compatible endpoint (OpenRouter, Groq,
/// Together, Ollama, llama.cpp, ...). Configuration comes from flags, then
/// `HARNESS_*`, then `OPENAI_*` environment variables; `.env` is loaded by the
/// binary.
pub struct OpenAiModel {
    client: llm_http::Client,
    model: String,
    max_tokens: Option<u32>,
}

impl OpenAiModel {
    pub fn new(base_url: &str, api_key: String, model: String) -> Self {
        let base_url = normalize_base_url(base_url);
        debug!(
            base_url = %base_url,
            model = %model,
            api_key_bytes = api_key.len(),
            "model configuration"
        );
        Self {
            client: llm_http::Client::new(&base_url, api_key),
            model,
            max_tokens: None,
        }
    }

    pub fn from_env(model: Option<&str>, base_url: Option<&str>) -> Result<Self> {
        let api_key = env::var("HARNESS_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .context(
                "no model API key found: set HARNESS_API_KEY (or OPENAI_API_KEY) \
                 in .env or the environment",
            )?;
        let base_url = base_url
            .map(str::to_string)
            .or_else(|| env::var("HARNESS_BASE_URL").ok())
            .or_else(|| env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = model
            .map(str::to_string)
            .or_else(|| env::var("HARNESS_MODEL").ok())
            .or_else(|| env::var("OPENAI_MODEL").ok())
            .context("no model configured: pass --model or set HARNESS_MODEL (or OPENAI_MODEL)")?;

        Ok(Self::new(&base_url, api_key, model))
    }

    pub fn base_url(&self) -> &str {
        self.client.base_url()
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = self.client.with_timeout(timeout);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    fn request(&self, messages: &[Message], tools: &[ToolDefinition]) -> ChatRequest {
        let mut request = ChatRequest::new(&self.model)
            .messages(messages.iter().map(to_llm_message))
            .tools(tools.iter().cloned());
        if let Some(max_tokens) = self.max_tokens {
            request = request.max_tokens(max_tokens);
        }
        request
    }
}

impl Model for OpenAiModel {
    fn name(&self) -> &str {
        &self.model
    }

    fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let request = self.request(messages, tools);

        debug!(
            model = %self.model,
            messages = messages.len(),
            tools = tools.len(),
            max_tokens = self.max_tokens,
            "calling model (streaming)"
        );

        let response = self
            .client
            .chat_stream(&request, |delta| on_delta(delta))
            .with_context(|| format!("calling model {}", self.model))?;

        debug!(
            finish = ?response.finish_reason,
            tool_calls = response.tool_calls.len(),
            total_tokens = response.usage.total_tokens,
            "model stream finished"
        );

        Ok(Completion {
            text: response.content.unwrap_or_default(),
            tool_calls: response.tool_calls,
            finish_reason: response.finish_reason,
            usage: response.usage,
        })
    }
}

/// Common configuration mistakes, corrected with a warning:
/// - pasting the full `/chat/completions` endpoint as the base URL;
/// - OpenRouter's `/api` or origin instead of `/api/v1`.
fn normalize_base_url(raw: &str) -> String {
    let mut base = raw.trim().trim_end_matches('/').to_string();

    if let Some(stripped) = base.strip_suffix("/chat/completions") {
        warn!(
            from = %base,
            "base URL includes the endpoint path; using the base instead"
        );
        base = stripped.trim_end_matches('/').to_string();
    }

    match base.as_str() {
        "https://openrouter.ai" | "http://openrouter.ai" => {
            warn!(from = %base, "OpenRouter base URL should be .../api/v1; correcting it");
            base.push_str("/api/v1");
        }
        "https://openrouter.ai/api" | "http://openrouter.ai/api" => {
            warn!(from = %base, "OpenRouter base URL is missing /v1; correcting it");
            base.push_str("/v1");
        }
        _ => {}
    }

    base
}

fn to_llm_message(message: &Message) -> llm_http::Message {
    match message.role {
        Role::System => llm_http::Message::system(message.content.clone()),
        Role::User => llm_http::Message::user(message.content.clone()),
        Role::Assistant if !message.tool_calls.is_empty() => {
            let mut converted = llm_http::Message::assistant_tool_calls(message.tool_calls.clone());
            if !message.content.is_empty() {
                converted.content = Some(message.content.clone());
            }
            converted
        }
        Role::Assistant => llm_http::Message::assistant(message.content.clone()),
        Role::Tool => llm_http::Message::tool_result(
            message.tool_call_id.clone().unwrap_or_default(),
            message.content.clone(),
        ),
    }
}

/// Replay canned assistant replies from a local file. This lets the loop run
/// end-to-end without a key: each call returns the next reply.
pub struct ReplayModel {
    name: String,
    replies: Mutex<VecDeque<String>>,
}

impl ReplayModel {
    pub fn new(replies: Vec<String>) -> Self {
        Self {
            name: "replay".to_string(),
            replies: Mutex::new(replies.into()),
        }
    }

    /// JSON array of reply strings, e.g. `["```rust\nfn main() {}```"]`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading replay file {}", path.display()))?;
        let replies: Vec<String> = serde_json::from_str(&text)
            .with_context(|| format!("parsing {} as a JSON array of strings", path.display()))?;
        let mut model = Self::new(replies);
        model.name = format!("replay:{}", path.display());
        Ok(model)
    }
}

impl Model for ReplayModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let text = self
            .replies
            .lock()
            .expect("replay lock")
            .pop_front()
            .context("replay file has no replies left")?;
        on_delta(&text);
        Ok(Completion {
            text,
            tool_calls: Vec::new(),
            finish_reason: Some("stop".to_string()),
            usage: Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_openrouter_urls() {
        assert_eq!(
            normalize_base_url("https://openrouter.ai"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn strips_endpoint_path_from_base_url() {
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn leaves_other_bases_alone() {
        assert_eq!(
            normalize_base_url("http://localhost:11434/v1/"),
            "http://localhost:11434/v1"
        );
    }

    #[test]
    fn converts_tool_messages() {
        let call = ToolCall::new("call_1", "read_file", "{}");
        let assistant = to_llm_message(&Message::assistant_tool_calls("", vec![call.clone()]));
        assert_eq!(assistant.tool_calls.len(), 1);
        assert!(assistant.content.is_none());

        let result = to_llm_message(&Message::tool_result("call_1", "contents"));
        assert_eq!(result.role, llm_http::Role::Tool);
        assert_eq!(result.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(result.content.as_deref(), Some("contents"));
    }
}
