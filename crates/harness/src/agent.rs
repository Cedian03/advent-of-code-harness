use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use llm_http::{ToolCall, ToolDefinition};
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::aoc::{Aoc, RunReport, Task, TaskId, TestReport, Verdict};
use crate::model::{Completion, Message, Model};
use crate::tools::{ToolContext, ToolRegistry};

const SYSTEM_PROMPT: &str = r#"You are an expert competitive programmer solving Advent of Code.

You work in a solution workspace. The program is `solve.rs`; write it with `write_file`, refine it with `edit_file`.

The program must:
- read the entire puzzle input from stdin;
- read the part number from the `AOC_PART` environment variable (1 or 2);
- print only the final answer to stdout.

Use the tools instead of pasting code into chat:
1. Read the statement and examples in the first message.
2. Use `read_input` to inspect the real input if that helps.
3. Write `solve.rs`.
4. Call `run_tests` and fix failures.
5. Call `run_part` for each captured part.
6. When done, stop calling tools and reply with a one-line summary.

Work in small steps. If a tool reports an error, read it and try again."#;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStart {
        turn: u32,
    },
    ModelStarted {
        turn: u32,
        prompt_bytes: usize,
    },
    ModelCompleted {
        turn: u32,
        ttft_ms: Option<u64>,
        duration_ms: u64,
        reply_bytes: usize,
        tool_calls: usize,
        total_tokens: u32,
    },
    ToolStarted {
        turn: u32,
        name: String,
    },
    ToolFinished {
        turn: u32,
        name: String,
        ok: bool,
        millis: u64,
        preview: String,
    },
    Evaluated {
        turn: u32,
        passed: u32,
        total: u32,
    },
    TurnFinished {
        turn: u32,
    },
    Failed {
        message: String,
    },
    Finished {
        turns: u32,
    },
}

#[derive(Debug, Clone, Copy)]
struct ChatMetrics {
    ttft: Option<Duration>,
    duration: Duration,
    bytes: usize,
    tool_calls: usize,
    total_tokens: u32,
}

impl ChatMetrics {
    fn record(
        &self,
        turn: u32,
        tests_passed: u32,
        tests_total: u32,
        tool_errors: u32,
    ) -> TurnMetrics {
        TurnMetrics {
            turn,
            ttft_ms: self.ttft.map(|value| value.as_millis() as u64),
            duration_ms: self.duration.as_millis() as u64,
            reply_bytes: self.bytes,
            tool_calls: self.tool_calls as u32,
            tool_errors,
            tests_passed,
            tests_total,
            total_tokens: self.total_tokens,
        }
    }
}

pub struct Agent<'a, A: Aoc, M: Model> {
    aoc: &'a A,
    model: &'a M,
    max_turns: u32,
    verbose: bool,
}

impl<'a, A: Aoc, M: Model> Agent<'a, A, M> {
    pub fn new(aoc: &'a A, model: &'a M, max_turns: u32) -> Self {
        Self {
            aoc,
            model,
            max_turns,
            verbose: true,
        }
    }

    pub fn quiet(mut self) -> Self {
        self.verbose = false;
        self
    }

    pub fn solve(&self, id: TaskId) -> Result<SolveReport> {
        self.solve_with(id, &mut |_| {})
    }

    pub fn solve_with(
        &self,
        id: TaskId,
        events: &mut dyn FnMut(&AgentEvent),
    ) -> Result<SolveReport> {
        let task = self.aoc.load(id)?;
        let workspace = self.aoc.workspace(id)?;
        fs::create_dir_all(&workspace)
            .with_context(|| format!("creating workspace {}", workspace.display()))?;

        debug!(
            task = %id,
            tests = task.tests.len(),
            parts = ?task.parts,
            input_bytes = task.input.as_ref().map(String::len),
            workspace = %workspace.display(),
            "loaded task"
        );

        let registry = ToolRegistry::default_set();
        let tools = registry.definitions();
        let tool_ctx = ToolContext {
            aoc: self.aoc,
            id,
            workspace,
            input: task.input.as_deref(),
        };

        let mut messages = vec![
            Message::system(SYSTEM_PROMPT),
            Message::user(initial_prompt(&task)),
        ];
        let mut history = Vec::new();
        let mut installed_by_tools = false;

        for turn in 1..=self.max_turns {
            events(&AgentEvent::TurnStart { turn });
            let (completion, metrics) = self.model_turn(turn, &messages, &tools, events)?;

            if completion.finish_reason.as_deref() == Some("length") {
                if self.verbose {
                    eprintln!("turn {turn}: reply hit the token limit");
                }
                messages.push(Message::assistant(completion.text));
                messages.push(Message::user(
                    "Your reply was cut off by the token limit. Use the tools to make a smaller \
                     edit, or finish the file with `write_file`.",
                ));
                history.push(metrics.record(turn, 0, 0, 0));
                events(&AgentEvent::TurnFinished { turn });
                continue;
            }

            if !completion.tool_calls.is_empty() {
                let (tool_errors, calls, wrote) = self.run_tools(
                    turn,
                    &completion,
                    &registry,
                    &tool_ctx,
                    &mut messages,
                    events,
                )?;
                installed_by_tools |= wrote;
                history.push(metrics.record(turn, 0, 0, tool_errors));
                debug!(turn, tool_calls = calls, "tool turn finished");
                events(&AgentEvent::TurnFinished { turn });
                continue;
            }

            // No tool calls: the model considers the task done. Verify. A full
            // code block in chat is accepted as a fallback (useful for models
            // that ignore the tool instructions); otherwise use solve.rs.
            // Prefer the workspace file once the model has written/edited it;
            // only accept a chat code block from models that ignored the tools.
            let fallback = if installed_by_tools {
                None
            } else {
                extract_code(&completion.text).filter(|code| code.contains("fn main"))
            };
            let tests = if let Some(code) = fallback {
                match self.aoc.evaluate(id, &code) {
                    Ok(report) => report,
                    Err(err) => {
                        messages.push(Message::assistant(completion.text));
                        messages.push(Message::user(format!(
                            "The solution failed to compile or run:\n\n```text\n{err:#}\n```\n\n\
                             Write the complete file with `write_file` and try again."
                        )));
                        history.push(metrics.record(turn, 0, 0, 0));
                        events(&AgentEvent::TurnFinished { turn });
                        continue;
                    }
                }
            } else {
                match self.aoc.evaluate_file(id) {
                    Ok(report) => report,
                    Err(err) => {
                        if self.verbose {
                            eprintln!("turn {turn}: no runnable solve.rs");
                        }
                        messages.push(Message::assistant(completion.text));
                        messages.push(Message::user(format!(
                            "There is no runnable `solve.rs` yet ({err:#}). Use `write_file` to \
                             create it, then call `run_tests`."
                        )));
                        history.push(metrics.record(turn, 0, 0, 0));
                        events(&AgentEvent::TurnFinished { turn });
                        continue;
                    }
                }
            };

            let passed = tests.passed() as u32;
            let total = tests.cases.len() as u32;
            events(&AgentEvent::Evaluated {
                turn,
                passed,
                total,
            });
            debug!(turn, passed, total, "evaluated solution");
            if self.verbose {
                eprintln!("turn {turn}: {passed}/{total} example test(s) passed");
            }
            history.push(metrics.record(turn, passed, total, 0));

            if tests.has_tests() && !tests.all_passed() {
                messages.push(Message::assistant(completion.text));
                messages.push(Message::user(test_feedback(&tests)));
                events(&AgentEvent::TurnFinished { turn });
                continue;
            }

            if task.input.is_none() {
                events(&AgentEvent::Finished { turns: turn });
                return Ok(SolveReport {
                    id,
                    turns: turn,
                    verified: tests.has_tests(),
                    tests,
                    runs: Vec::new(),
                    recorded: Vec::new(),
                    history,
                    note: Some("no puzzle input available; ran example tests only".to_string()),
                });
            }

            let mut runs = Vec::new();
            for &part in &task.parts {
                let run = self.aoc.run_file(id, part)?;
                if self.verbose {
                    eprintln!(
                        "turn {turn}: part {part} -> {}{}",
                        run.answer,
                        match run.verdict {
                            Verdict::Correct => " (matches archive)",
                            Verdict::Incorrect => " (WRONG, differs from archive)",
                            Verdict::Unverified => "",
                        }
                    );
                }
                runs.push(run);
            }

            if runs.iter().any(|run| !run.success) {
                messages.push(Message::assistant(completion.text));
                messages.push(Message::user(run_failure_feedback(&runs)));
                events(&AgentEvent::TurnFinished { turn });
                continue;
            }

            if runs.iter().any(|run| run.verdict == Verdict::Incorrect) {
                messages.push(Message::assistant(completion.text));
                messages.push(Message::user(format!(
                    "The example tests pass, but the real input contradicts an answer AoC \
                     already recorded for this account:\n\n{}\n\n\
                     Fix the logic, call `run_tests` and `run_part` again.",
                    runs.iter()
                        .filter(|run| run.verdict == Verdict::Incorrect)
                        .map(|run| format!("- part {}: got `{}`", run.part, run.answer))
                        .collect::<Vec<_>>()
                        .join("\n")
                )));
                events(&AgentEvent::TurnFinished { turn });
                continue;
            }

            for run in &runs {
                self.aoc.submit(id, run.part, &run.answer)?;
            }

            events(&AgentEvent::Finished { turns: turn });
            return Ok(SolveReport {
                id,
                turns: turn,
                verified: tests.has_tests(),
                tests,
                runs,
                recorded: task.parts.clone(),
                history,
                note: None,
            });
        }

        let message = format!("no passing solution after {} turn(s)", self.max_turns);
        events(&AgentEvent::Failed {
            message: message.clone(),
        });
        bail!(message)
    }

    fn run_tools(
        &self,
        turn: u32,
        completion: &Completion,
        registry: &ToolRegistry,
        ctx: &ToolContext<'_>,
        messages: &mut Vec<Message>,
        events: &mut dyn FnMut(&AgentEvent),
    ) -> Result<(u32, usize, bool)> {
        let mut calls = completion.tool_calls.clone();
        for (index, call) in calls.iter_mut().enumerate() {
            if call.id.is_empty() {
                call.id = format!("call_{turn}_{index}");
            }
        }
        messages.push(Message::assistant_tool_calls(
            completion.text.clone(),
            calls.clone(),
        ));

        let mut errors = 0;
        let mut wrote = false;
        for call in &calls {
            events(&AgentEvent::ToolStarted {
                turn,
                name: call.function.name.clone(),
            });
            if self.verbose {
                eprintln!("turn {turn}: tool {}", call.function.name);
            }

            let started = Instant::now();
            let result = execute_tool(registry, ctx, call);
            let millis = started.elapsed().as_millis() as u64;
            let (ok, content) = match result {
                Ok(content) => (true, content),
                Err(err) => {
                    errors += 1;
                    (false, format!("ERROR: {err:#}"))
                }
            };
            if ok && matches!(call.function.name.as_str(), "write_file" | "edit_file") {
                wrote = true;
            }
            debug!(turn, tool = %call.function.name, ok, millis, "tool finished");
            events(&AgentEvent::ToolFinished {
                turn,
                name: call.function.name.clone(),
                ok,
                millis,
                preview: preview(&content),
            });
            messages.push(Message::tool_result(call.id.clone(), content));
        }

        Ok((errors, calls.len(), wrote))
    }

    /// One model call, with a heartbeat so a slow provider does not look hung.
    fn model_turn(
        &self,
        turn: u32,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &mut dyn FnMut(&AgentEvent),
    ) -> Result<(Completion, ChatMetrics)> {
        let prompt_bytes: usize = messages.iter().map(|message| message.content.len()).sum();
        if self.verbose {
            eprintln!(
                "turn {turn}: asking {} ({:.1} KB prompt)",
                self.model.name(),
                prompt_bytes as f64 / 1024.0
            );
        }
        events(&AgentEvent::ModelStarted { turn, prompt_bytes });

        let started = Instant::now();
        let streaming = Arc::new(AtomicBool::new(false));
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let watchdog = self.verbose.then(|| {
            let name = self.model.name().to_string();
            let streaming = Arc::clone(&streaming);
            thread::spawn(move || {
                let start = Instant::now();
                loop {
                    match stop_rx.recv_timeout(Duration::from_secs(10)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if streaming.load(Ordering::Relaxed) {
                                break;
                            }
                            eprintln!(
                                "turn {turn}: waiting for {name}... {}s",
                                start.elapsed().as_secs()
                            );
                        }
                    }
                }
            })
        });

        let mut first_token = None;
        let mut received = 0usize;
        let mut last_report = Instant::now();
        let mut on_delta = |delta: &str| {
            if first_token.is_none() {
                first_token = Some(started.elapsed());
                streaming.store(true, Ordering::Relaxed);
                if self.verbose {
                    eprintln!(
                        "turn {turn}: first token after {:.1}s",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            received += delta.len();
            if self.verbose && last_report.elapsed() >= Duration::from_secs(3) {
                eprintln!(
                    "turn {turn}: generating... {:.1} KB",
                    received as f64 / 1024.0
                );
                last_report = Instant::now();
            }
        };

        let result = self.model.complete(messages, tools, &mut on_delta);
        let _ = stop_tx.send(());
        if let Some(handle) = watchdog {
            let _ = handle.join();
        }

        let tool_bytes = result
            .as_ref()
            .map(|completion| {
                completion
                    .tool_calls
                    .iter()
                    .map(|call| call.function.name.len() + call.function.arguments.len())
                    .sum::<usize>()
            })
            .unwrap_or(0);
        let metrics = ChatMetrics {
            ttft: first_token,
            duration: started.elapsed(),
            bytes: received + tool_bytes,
            tool_calls: result.as_ref().map(|c| c.tool_calls.len()).unwrap_or(0),
            total_tokens: result.as_ref().map(|c| c.usage.total_tokens).unwrap_or(0),
        };

        match &result {
            Ok(completion) => {
                info!(
                    turn,
                    ttft_ms = metrics
                        .ttft
                        .map(|value| value.as_millis() as u64)
                        .unwrap_or(0),
                    duration_ms = metrics.duration.as_millis() as u64,
                    reply_bytes = metrics.bytes,
                    tool_calls = completion.tool_calls.len(),
                    total_tokens = completion.usage.total_tokens,
                    "model turn"
                );
                events(&AgentEvent::ModelCompleted {
                    turn,
                    ttft_ms: metrics.ttft.map(|value| value.as_millis() as u64),
                    duration_ms: metrics.duration.as_millis() as u64,
                    reply_bytes: metrics.bytes,
                    tool_calls: completion.tool_calls.len(),
                    total_tokens: completion.usage.total_tokens,
                });
            }
            Err(err) => warn!(
                turn,
                duration_ms = metrics.duration.as_millis() as u64,
                error = %err,
                "model call failed"
            ),
        }

        if self.verbose {
            let seconds = metrics.duration.as_secs_f64();
            match &result {
                Ok(completion) => eprintln!(
                    "turn {turn}: model replied in {seconds:.1}s ({:.1} KB, {} tool call(s))",
                    received as f64 / 1024.0,
                    completion.tool_calls.len()
                ),
                Err(_) => eprintln!("turn {turn}: model call failed after {seconds:.1}s"),
            }
        }

        let completion = result?;
        Ok((completion, metrics))
    }
}

fn execute_tool(registry: &ToolRegistry, ctx: &ToolContext<'_>, call: &ToolCall) -> Result<String> {
    let arguments = call.function.arguments.trim();
    let args: Value = if arguments.is_empty() {
        Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str(arguments)
            .with_context(|| format!("invalid JSON arguments for `{}`", call.function.name))?
    };
    registry.execute(ctx, &call.function.name, args)
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnMetrics {
    pub turn: u32,
    /// Time from request start to the first streamed token.
    pub ttft_ms: Option<u64>,
    pub duration_ms: u64,
    pub reply_bytes: usize,
    pub tool_calls: u32,
    pub tool_errors: u32,
    pub tests_passed: u32,
    pub tests_total: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolveReport {
    pub id: TaskId,
    pub turns: u32,
    /// True when at least one example test verified the solution.
    pub verified: bool,
    pub tests: TestReport,
    pub runs: Vec<RunReport>,
    /// Parts whose answers were recorded locally.
    pub recorded: Vec<u8>,
    /// Per-turn model timings (TTFT, total duration, test results).
    pub history: Vec<TurnMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

fn initial_prompt(task: &Task) -> String {
    let mut prompt = format!(
        "# {} Day {}: {}\n\n{}\n\n---\n\n",
        task.id.year, task.id.day, task.title, task.statement
    );

    if task.tests.is_empty() {
        prompt.push_str(
            "No example tests were detected for this day. Derive the rules carefully \
             from the statement.\n\n",
        );
    } else {
        prompt.push_str("## Example tests\n\n");
        for (index, test) in task.tests.iter().enumerate() {
            prompt.push_str(&format!(
                "Example {} (part {}):\n\n```\n{}\n```\n\nExpected output: `{}`\n\n",
                index + 1,
                test.part,
                test.input.trim_end_matches('\n'),
                test.expected
            ));
        }
    }

    match &task.input {
        Some(input) => prompt.push_str(&format!(
            "The real input ({} bytes) is available to your program on stdin and to you via \
             `read_input`. Handle {} in the same file.",
            input.len(),
            if task.parts.len() > 1 {
                "both parts"
            } else {
                "the captured part"
            }
        )),
        None => prompt.push_str(
            "No real input is available locally; only the examples can be checked by the \
             harness.",
        ),
    }

    prompt
}

fn test_feedback(tests: &TestReport) -> String {
    let mut feedback = String::from("The example tests did not pass:\n\n");
    feedback.push_str(&crate::tools::format_tests(tests));
    feedback.push_str("\nFix `solve.rs`, then call `run_tests` again.");
    feedback
}

fn run_failure_feedback(runs: &[RunReport]) -> String {
    let mut feedback = String::from("The solution failed on the real input:\n\n");
    for run in runs.iter().filter(|run| !run.success) {
        feedback.push_str(&format!(
            "- part {}: process exited unsuccessfully\n",
            run.part
        ));
        for line in run.stderr.trim().lines().take(8) {
            feedback.push_str(&format!("    {line}\n"));
        }
    }
    feedback.push_str("\nFix `solve.rs`, then call `run_tests` and `run_part` again.");
    feedback
}

fn preview(text: &str) -> String {
    const LIMIT: usize = 200;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= LIMIT {
        return collapsed;
    }
    let mut out: String = collapsed.chars().take(LIMIT).collect();
    out.push('…');
    out
}

/// Pull the last Rust (or unlabelled) fenced block out of a model reply.
pub fn extract_code(reply: &str) -> Option<String> {
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut lines = reply.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(info) = trimmed.strip_prefix("```") else {
            continue;
        };
        let language = info.trim().to_lowercase();
        let mut body = String::new();
        for line in lines.by_ref() {
            if line.trim_start().starts_with("```") {
                break;
            }
            body.push_str(line);
            body.push('\n');
        }
        blocks.push((language, body));
    }

    blocks
        .iter()
        .rev()
        .find(|(language, _)| {
            language.is_empty() || language.starts_with("rust") || language == "rs"
        })
        .map(|(_, body)| body.trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aoc::{CaseResult, TaskSummary};
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct MockAoc {
        workspace: PathBuf,
    }

    impl MockAoc {
        fn new(name: &str) -> Self {
            Self {
                workspace: std::env::temp_dir()
                    .join(format!("harness-agent-test-{}-{name}", std::process::id())),
            }
        }
    }

    impl Drop for MockAoc {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.workspace);
        }
    }

    impl Aoc for MockAoc {
        fn workspace(&self, _id: TaskId) -> Result<PathBuf> {
            Ok(self.workspace.clone())
        }

        fn tasks(&self) -> Result<Vec<TaskSummary>> {
            Ok(Vec::new())
        }

        fn load(&self, id: TaskId) -> Result<Task> {
            Ok(Task {
                id,
                title: "Test Day".to_string(),
                url: "https://example.com".to_string(),
                statement: "Add one.".to_string(),
                tests: vec![crate::aoc::TestCase {
                    part: 1,
                    input: "3\n".to_string(),
                    expected: "4".to_string(),
                }],
                input: Some("5\n".to_string()),
                parts: vec![1],
            })
        }

        fn evaluate(&self, id: TaskId, source: &str) -> Result<TestReport> {
            let workspace = self.workspace(id)?;
            std::fs::create_dir_all(&workspace)?;
            std::fs::write(workspace.join("solve.rs"), source)?;
            let passed = source.contains("GOOD");
            Ok(TestReport {
                cases: vec![CaseResult {
                    part: 1,
                    expected: "4".to_string(),
                    actual: if passed { "4" } else { "0" }.to_string(),
                    passed,
                    stderr: String::new(),
                    millis: 0,
                }],
            })
        }

        fn run(&self, _id: TaskId, part: u8, _source: &str) -> Result<RunReport> {
            Ok(RunReport {
                part,
                answer: "4".to_string(),
                stderr: String::new(),
                success: true,
                millis: 0,
                verdict: Verdict::Unverified,
            })
        }

        fn submit(&self, _id: TaskId, _part: u8, _answer: &str) -> Result<Verdict> {
            Ok(Verdict::Unverified)
        }
    }

    struct ScriptedModel {
        replies: Mutex<Vec<Completion>>,
    }

    impl ScriptedModel {
        fn new(replies: Vec<Completion>) -> Self {
            Self {
                replies: Mutex::new(replies),
            }
        }
    }

    impl Model for ScriptedModel {
        fn name(&self) -> &str {
            "scripted"
        }

        fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            on_delta: &mut dyn FnMut(&str),
        ) -> Result<Completion> {
            let mut replies = self.replies.lock().unwrap();
            if replies.is_empty() {
                bail!("script exhausted");
            }
            let completion = replies.remove(0);
            on_delta(&completion.text);
            Ok(completion)
        }
    }

    fn text_completion(text: &str) -> Completion {
        Completion {
            text: text.to_string(),
            tool_calls: Vec::new(),
            finish_reason: Some("stop".to_string()),
            usage: llm_http::Usage::default(),
        }
    }

    #[test]
    fn extracts_last_rust_block() {
        let reply = "First:\n```rust\nfn main() { println!(\"one\"); }\n```\n\
                     Fixed:\n```rust\nfn main() { println!(\"two\"); }\n```\n";
        let code = extract_code(reply).unwrap();
        assert!(code.contains("two"));
        assert!(!code.contains("one"));
    }

    #[test]
    fn ignores_non_code_replies() {
        assert!(extract_code("Here is an explanation.").is_none());
    }

    #[test]
    fn loop_retries_after_failed_tests() {
        let aoc = MockAoc::new("retry");
        let model = ScriptedModel::new(vec![
            text_completion("```rust\nfn main() { let _ = \"BAD\"; }\n```"),
            text_completion("```rust\nfn main() { let _ = \"GOOD\"; }\n```"),
        ]);
        let report = Agent::new(&aoc, &model, 4)
            .quiet()
            .solve(TaskId::new(2024, 1))
            .unwrap();

        assert_eq!(report.turns, 2);
        assert!(report.verified);
        assert_eq!(report.runs[0].answer, "4");
    }

    #[test]
    fn executes_tool_calls_and_then_evaluates() {
        let aoc = MockAoc::new("tools");
        let write = Completion {
            text: String::new(),
            tool_calls: vec![llm_http::ToolCall::new(
                "call_1",
                "write_file",
                serde_json::json!({"path": "solve.rs", "content": "GOOD"}).to_string(),
            )],
            finish_reason: Some("tool_calls".to_string()),
            usage: llm_http::Usage::default(),
        };
        let model = ScriptedModel::new(vec![write, text_completion("done")]);
        let report = Agent::new(&aoc, &model, 4)
            .quiet()
            .solve(TaskId::new(2024, 1))
            .unwrap();

        assert_eq!(report.turns, 2);
        assert_eq!(report.history[0].tool_calls, 1);
        assert_eq!(report.history[1].tests_passed, 1);
    }
}
