use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use llm_http::ToolDefinition;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::aoc::{Aoc, TaskId, TestReport, Verdict};

const MAX_FILE_CHARS: usize = 40_000;
const MAX_INPUT_CHARS: usize = 12_000;
const MAX_TOOL_RESULT_CHARS: usize = 20_000;

pub struct ToolContext<'a> {
    pub aoc: &'a dyn Aoc,
    pub id: TaskId,
    pub workspace: PathBuf,
    pub input: Option<&'a str>,
}

pub trait Tool {
    fn name(&self) -> &'static str;
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String>;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn default_set() -> Self {
        Self {
            tools: vec![
                Box::new(ReadFile),
                Box::new(WriteFile),
                Box::new(EditFile),
                Box::new(ListFiles),
                Box::new(ReadInput),
                Box::new(RunTests),
                Box::new(RunPart),
            ],
        }
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }

    pub fn execute(&self, ctx: &ToolContext<'_>, name: &str, args: Value) -> Result<String> {
        let Some(tool) = self.tools.iter().find(|tool| tool.name() == name) else {
            bail!("unknown tool `{name}`");
        };
        let result = tool.execute(ctx, args)?;
        Ok(truncate(result, MAX_TOOL_RESULT_CHARS))
    }
}

struct ReadFile;

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
}

impl Tool for ReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Read a text file from the solution workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace, e.g. solve.rs"}
                },
                "required": ["path"]
            }),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String> {
        let args: ReadArgs = serde_json::from_value(args).context("invalid read_file arguments")?;
        let path = resolve(&ctx.workspace, &args.path)?;
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(format!(
            "{} ({} bytes, {} lines)\n\n{}",
            args.path,
            text.len(),
            text.lines().count(),
            truncate(text, MAX_FILE_CHARS)
        ))
    }
}

struct WriteFile;

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

impl Tool for WriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Create or overwrite a file in the solution workspace. Use this to write solve.rs.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace, usually solve.rs"},
                    "content": {"type": "string", "description": "Complete file contents"}
                },
                "required": ["path", "content"]
            }),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String> {
        let args: WriteArgs =
            serde_json::from_value(args).context("invalid write_file arguments")?;
        let path = resolve(&ctx.workspace, &args.path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, &args.content).with_context(|| format!("writing {}", path.display()))?;
        Ok(format!(
            "wrote {} bytes ({} lines) to {}",
            args.content.len(),
            args.content.lines().count(),
            args.path
        ))
    }
}

struct EditFile;

#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
    #[serde(default)]
    replace_all: bool,
}

impl Tool for EditFile {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Replace exact text in a file. `old` must match exactly one location unless replace_all is true.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace"},
                    "new": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "default": false}
                },
                "required": ["path", "old", "new"]
            }),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String> {
        let args: EditArgs = serde_json::from_value(args).context("invalid edit_file arguments")?;
        let path = resolve(&ctx.workspace, &args.path)?;
        let source =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let occurrences = source.matches(&args.old).count();
        if occurrences == 0 {
            bail!("`old` text was not found in {}", args.path);
        }
        if occurrences > 1 && !args.replace_all {
            bail!("`old` text appears {occurrences} times; add more context or set replace_all");
        }
        let updated = if args.replace_all {
            source.replace(&args.old, &args.new)
        } else {
            source.replacen(&args.old, &args.new, 1)
        };
        fs::write(&path, updated).with_context(|| format!("writing {}", path.display()))?;
        Ok(format!(
            "replaced {occurrences} occurrence(s) in {}",
            args.path
        ))
    }
}

struct ListFiles;

#[derive(Deserialize, Default)]
struct ListArgs {
    #[serde(default = "dot")]
    path: String,
}

fn dot() -> String {
    ".".to_string()
}

impl Tool for ListFiles {
    fn name(&self) -> &'static str {
        "list_files"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "List files in the solution workspace (one directory level).",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."}
                }
            }),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String> {
        let args: ListArgs = serde_json::from_value(args).unwrap_or_default();
        let dir = resolve(&ctx.workspace, &args.path)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(&dir).with_context(|| format!("listing {}", dir.display()))? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let kind = if file_type.is_dir() { "dir " } else { "file" };
            let size = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
            entries.push(format!(
                "{kind} {:>8} {}",
                size,
                entry.file_name().to_string_lossy()
            ));
        }
        entries.sort();
        if entries.is_empty() {
            return Ok(format!("{} is empty", args.path));
        }
        Ok(entries.join("\n"))
    }
}

struct ReadInput;

impl Tool for ReadInput {
    fn name(&self) -> &'static str {
        "read_input"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Read your personal puzzle input. It may be truncated; the solution reads the full input from stdin.",
            json!({"type": "object", "properties": {}}),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, _args: Value) -> Result<String> {
        let Some(input) = ctx.input else {
            bail!("no puzzle input is available for this task; import it with `harness input`");
        };
        Ok(format!(
            "input ({} bytes, {} lines)\n\n{}",
            input.len(),
            input.lines().count(),
            truncate(input.to_string(), MAX_INPUT_CHARS)
        ))
    }
}

struct RunTests;

impl Tool for RunTests {
    fn name(&self) -> &'static str {
        "run_tests"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Compile solve.rs and run the detected example tests.",
            json!({"type": "object", "properties": {}}),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, _args: Value) -> Result<String> {
        let report = ctx.aoc.evaluate_file(ctx.id)?;
        Ok(format_tests(&report))
    }
}

struct RunPart;

#[derive(Deserialize)]
struct RunPartArgs {
    #[serde(default = "one")]
    part: u8,
}

fn one() -> u8 {
    1
}

impl Tool for RunPart {
    fn name(&self) -> &'static str {
        "run_part"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            self.name(),
            "Run solve.rs on the real puzzle input for part 1 or 2 and return the answer.",
            json!({
                "type": "object",
                "properties": {
                    "part": {"type": "integer", "enum": [1, 2], "default": 1}
                }
            }),
        )
    }

    fn execute(&self, ctx: &ToolContext<'_>, args: Value) -> Result<String> {
        let args: RunPartArgs = serde_json::from_value(args).unwrap_or(RunPartArgs { part: 1 });
        if !(1..=2).contains(&args.part) {
            bail!("part must be 1 or 2");
        }
        let report = ctx.aoc.run_file(ctx.id, args.part)?;
        let mut out = format!(
            "part {}: `{}` ({} ms, {})",
            report.part,
            report.answer,
            report.millis,
            verdict_label(report.verdict)
        );
        if !report.success {
            out.push_str("\nprocess failed");
        }
        if !report.stderr.trim().is_empty() {
            out.push_str("\nstderr:\n");
            for line in report.stderr.trim().lines().take(8) {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(out)
    }
}

pub fn format_tests(report: &TestReport) -> String {
    if report.cases.is_empty() {
        return "No example tests were detected for this day; use run_part to check the real input."
            .to_string();
    }

    let mut out = format!(
        "{}/{} example tests passed\n",
        report.passed(),
        report.cases.len()
    );
    for case in &report.cases {
        if case.passed {
            continue;
        }
        out.push_str(&format!(
            "- part {}: expected `{}`, got `{}`\n",
            case.part,
            case.expected,
            if case.actual.is_empty() {
                "<no output>"
            } else {
                &case.actual
            }
        ));
        if !case.stderr.trim().is_empty() {
            out.push_str("  stderr:\n");
            for line in case.stderr.trim().lines().take(6) {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

fn verdict_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Correct => "matches archived answer",
        Verdict::Incorrect => "does NOT match archived answer",
        Verdict::Unverified => "recorded, no archived answer",
    }
}

fn resolve(workspace: &Path, raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    if path.is_absolute() {
        bail!("path must be relative to the solution workspace");
    }
    for component in path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("path may not contain `..` or a prefix");
            }
        }
    }
    Ok(workspace.join(path))
}

fn truncate(text: String, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text;
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push_str(&format!("\n… ({} chars truncated)", count - limit));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_outside_the_workspace() {
        let workspace = Path::new("/tmp/ws");
        assert!(resolve(workspace, "../secret").is_err());
        assert!(resolve(workspace, "/etc/passwd").is_err());
        assert!(resolve(workspace, "solve.rs").is_ok());
        assert!(resolve(workspace, "src/main.rs").is_ok());
    }

    #[test]
    fn truncates_on_char_boundaries() {
        let text = "αβγδε".to_string();
        let out = truncate(text, 2);
        assert!(out.starts_with("αβ"));
        assert!(out.contains("3 chars truncated"));
    }
}
