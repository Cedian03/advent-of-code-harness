//! A small, synchronous client for the OpenAI-compatible Chat Completions
//! protocol.
//!
//! It deliberately covers one endpoint: `POST {base_url}/chat/completions`.
//! Messages, tool definitions, tool calls, and usage are typed; anything else
//! is left to the caller. Works with OpenAI, OpenRouter, Groq, Together,
//! Ollama, llama.cpp, and any other compatible server.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP {status} from {url}: {body}")]
    Http {
        status: u16,
        url: String,
        body: String,
    },

    #[error("request to {url} failed: {source}")]
    Transport {
        url: String,
        #[source]
        source: ureq::Error,
    },

    #[error("reading stream from {url}: {source}")]
    Stream {
        url: String,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid response from {url}: {source}")]
    Decode {
        url: String,
        body: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("no choices in response from {url}: {body}")]
    EmptyChoices { url: String, body: String },

    #[error("could not encode request: {0}")]
    Encode(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

fn function_kind() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, exactly as returned by the model.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }

    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// `parameters` must be a JSON Schema object.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            tool_choice: None,
            stream: None,
        }
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn text(&self) -> Option<&str> {
        self.content.as_deref()
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCall>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunction>,
}

#[derive(Deserialize, Default)]
struct StreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn accumulate_tool_call(partials: &mut BTreeMap<usize, PartialToolCall>, delta: StreamToolCall) {
    let partial = partials.entry(delta.index).or_default();
    if let Some(id) = delta.id {
        partial.id = id;
    }
    if let Some(function) = delta.function {
        if let Some(name) = function.name {
            partial.name.push_str(&name);
        }
        if let Some(arguments) = function.arguments {
            partial.arguments.push_str(&arguments);
        }
    }
}

#[derive(Deserialize)]
struct RawResponse {
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct RawChoice {
    message: RawMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

pub struct Client {
    agent: ureq::Agent,
    base_url: String,
    url: String,
    api_key: String,
    headers: Vec<(String, String)>,
    timeout: Duration,
}

impl Client {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let base_url = base_url.trim_end_matches('/').to_string();
        let url = format!("{base_url}/chat/completions");
        let timeout = DEFAULT_TIMEOUT;
        Self {
            agent: build_agent(timeout),
            base_url,
            url,
            api_key: api_key.into(),
            headers: Vec::new(),
            timeout,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.agent = build_agent(timeout);
        self
    }

    /// Extra headers, e.g. OpenRouter's `HTTP-Referer` / `X-Title`.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, Error> {
        let body = serde_json::to_string(request)?;
        let mut response = self.post(body)?;

        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|source| Error::Transport {
                url: self.url.clone(),
                source,
            })?;

        if !(200..300).contains(&status) {
            return Err(Error::Http {
                status,
                url: self.url.clone(),
                body: snippet(&text),
            });
        }

        let raw: RawResponse = serde_json::from_str(&text).map_err(|source| Error::Decode {
            url: self.url.clone(),
            body: snippet(&text),
            source,
        })?;

        let choice = raw
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::EmptyChoices {
                url: self.url.clone(),
                body: snippet(&text),
            })?;

        Ok(ChatResponse {
            content: choice.message.content,
            tool_calls: choice.message.tool_calls,
            finish_reason: choice.finish_reason,
            usage: raw.usage,
        })
    }

    /// Stream a completion over SSE, calling `on_delta` for each text chunk.
    ///
    /// This makes time-to-first-token observable and lets a caller show
    /// progress while a slow provider is generating. Tool-call deltas are not
    /// assembled yet; only text content is returned.
    pub fn chat_stream<F>(
        &self,
        request: &ChatRequest,
        mut on_delta: F,
    ) -> Result<ChatResponse, Error>
    where
        F: FnMut(&str),
    {
        let mut request = request.clone();
        request.stream = Some(true);
        let body = serde_json::to_string(&request)?;
        let mut response = self.post(body)?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let text = response
                .body_mut()
                .read_to_string()
                .map_err(|source| Error::Transport {
                    url: self.url.clone(),
                    source,
                })?;
            return Err(Error::Http {
                status,
                url: self.url.clone(),
                body: snippet(&text),
            });
        }

        // Some compatible servers ignore `stream: true` and return ordinary
        // JSON. Fall back to the non-streaming shape instead of returning "".
        let is_event_stream = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains("text/event-stream"));

        if !is_event_stream {
            let text = response
                .body_mut()
                .read_to_string()
                .map_err(|source| Error::Transport {
                    url: self.url.clone(),
                    source,
                })?;
            let raw: RawResponse = serde_json::from_str(&text).map_err(|source| Error::Decode {
                url: self.url.clone(),
                body: snippet(&text),
                source,
            })?;
            let choice = raw
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| Error::EmptyChoices {
                    url: self.url.clone(),
                    body: snippet(&text),
                })?;
            if let Some(content) = &choice.message.content {
                on_delta(content);
            }
            return Ok(ChatResponse {
                content: choice.message.content,
                tool_calls: choice.message.tool_calls,
                finish_reason: choice.finish_reason,
                usage: raw.usage,
            });
        }

        let reader = response.body_mut().as_reader();
        let lines = BufReader::new(reader).lines();
        let mut content = String::new();
        let mut finish_reason = None;
        let mut usage = Usage::default();
        let mut partial_tool_calls: BTreeMap<usize, PartialToolCall> = BTreeMap::new();

        for line in lines {
            let line = line.map_err(|source| Error::Stream {
                url: self.url.clone(),
                source,
            })?;
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                break;
            }

            let chunk: StreamChunk =
                serde_json::from_str(data).map_err(|source| Error::Decode {
                    url: self.url.clone(),
                    body: snippet(data),
                    source,
                })?;

            if let Some(value) = chunk.usage {
                usage = value;
            }
            if let Some(choice) = chunk.choices.into_iter().next() {
                if let Some(delta) = choice.delta.content {
                    content.push_str(&delta);
                    on_delta(&delta);
                }
                for call in choice.delta.tool_calls {
                    accumulate_tool_call(&mut partial_tool_calls, call);
                }
                if choice.finish_reason.is_some() {
                    finish_reason = choice.finish_reason;
                }
            }
        }

        let tool_calls = partial_tool_calls
            .into_values()
            .filter(|call| !call.name.is_empty())
            .map(|call| ToolCall::new(call.id, call.name, call.arguments))
            .collect();

        Ok(ChatResponse {
            content: (!content.is_empty()).then_some(content),
            tool_calls,
            finish_reason,
            usage,
        })
    }

    fn post(&self, body: String) -> Result<ureq::http::Response<ureq::Body>, Error> {
        let mut call = self
            .agent
            .post(&self.url)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        for (name, value) in &self.headers {
            call = call.header(name, value);
        }

        call.send(body).map_err(|source| Error::Transport {
            url: self.url.clone(),
            source,
        })
    }
}

fn build_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

fn snippet(text: &str) -> String {
    const LIMIT: usize = 800;
    let text = text.trim();
    if text.chars().count() <= LIMIT {
        return text.to_string();
    }
    let mut out: String = text.chars().take(LIMIT).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_plain_message() {
        let json = serde_json::to_value(Message::user("hi")).unwrap();
        assert_eq!(json, serde_json::json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn serializes_tool_result() {
        let json = serde_json::to_value(Message::tool_result("call_1", "42")).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "42"})
        );
    }

    #[test]
    fn serializes_tool_definition_and_calls() {
        let tool = ToolDefinition::function(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        );
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "read_file");

        let message = Message::assistant_tool_calls(vec![ToolCall::new(
            "call_1",
            "read_file",
            r#"{"path":"input.txt"}"#,
        )]);
        let json = serde_json::to_value(message).unwrap();
        assert_eq!(json["role"], "assistant");
        assert!(!json.as_object().unwrap().contains_key("content"));
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn accumulates_streamed_tool_call_arguments() {
        let mut partials = BTreeMap::new();
        let first: StreamToolCall = serde_json::from_value(serde_json::json!({
            "index": 0,
            "id": "call_1",
            "function": {"name": "write_file", "arguments": "{\"path\":"}
        }))
        .unwrap();
        let second: StreamToolCall = serde_json::from_value(serde_json::json!({
            "index": 0,
            "function": {"arguments": "\"solve.rs\"}"}
        }))
        .unwrap();
        accumulate_tool_call(&mut partials, first);
        accumulate_tool_call(&mut partials, second);

        let call = partials.remove(&0).unwrap();
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "write_file");
        assert_eq!(call.arguments, r#"{"path":"solve.rs"}"#);
    }

    #[test]
    fn parses_stream_delta() {
        let chunk: StreamChunk = serde_json::from_str(
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn parses_response_with_tool_calls() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let raw: RawResponse = serde_json::from_value(body).unwrap();
        let choice = raw.choices.into_iter().next().unwrap();
        assert_eq!(choice.message.tool_calls.len(), 1);
        assert_eq!(choice.message.tool_calls[0].function.name, "run");
        assert_eq!(raw.usage.total_tokens, 15);
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }
}
