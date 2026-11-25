use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use aoc_archive::model::Puzzle;
use aoc_archive::scrape;

use crate::runner;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId {
    pub year: u32,
    pub day: u32,
}

impl TaskId {
    pub fn new(year: u32, day: u32) -> Self {
        Self { year, day }
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{:02}", self.year, self.day)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TestCase {
    pub part: u8,
    pub input: String,
    pub expected: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub id: TaskId,
    pub title: String,
    pub url: String,
    pub statement: String,
    pub tests: Vec<TestCase>,
    /// Your personal input, when it has been imported or scraped.
    pub input: Option<String>,
    /// Captured parts, usually [1] on a public archive.
    pub parts: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSummary {
    pub id: TaskId,
    pub title: String,
    pub tests: u32,
    pub parts: u8,
    pub has_input: bool,
    /// Parts already answered and recorded.
    pub solved: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub part: u8,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
    pub stderr: String,
    pub millis: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestReport {
    pub cases: Vec<CaseResult>,
}

impl TestReport {
    pub fn has_tests(&self) -> bool {
        !self.cases.is_empty()
    }

    pub fn all_passed(&self) -> bool {
        self.cases.iter().all(|case| case.passed)
    }

    pub fn passed(&self) -> usize {
        self.cases.iter().filter(|case| case.passed).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Matches an answer AoC recorded for this account.
    Correct,
    /// Contradicts a known answer.
    Incorrect,
    /// No oracle: the answer was recorded but not checked.
    Unverified,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub part: u8,
    pub answer: String,
    pub stderr: String,
    pub success: bool,
    pub millis: u64,
    pub verdict: Verdict,
}

/// Everything the solving loop needs from Advent of Code.
///
/// The loop never touches the archive, the filesystem, or the site directly;
/// it only talks to this trait. Implementations may be local (archive-backed)
/// or remote (a live submission client).
pub trait Aoc {
    /// All captured tasks, newest year last.
    fn tasks(&self) -> Result<Vec<TaskSummary>>;

    /// Load the full statement, detected examples, and input for one day.
    fn load(&self, id: TaskId) -> Result<Task>;

    /// Directory where the agent writes `solve.rs` for this task.
    fn workspace(&self, id: TaskId) -> Result<PathBuf>;

    /// Install a candidate Rust source and run the detected example tests.
    fn evaluate(&self, id: TaskId, source: &str) -> Result<TestReport>;

    /// Evaluate the `solve.rs` currently installed in the workspace.
    fn evaluate_file(&self, id: TaskId) -> Result<TestReport> {
        let path = self.workspace(id)?.join("solve.rs");
        let source = fs::read_to_string(&path)
            .with_context(|| format!("no solution at {}", path.display()))?;
        self.evaluate(id, &source)
    }

    /// Run a candidate source against the real input for one part.
    fn run(&self, id: TaskId, part: u8, source: &str) -> Result<RunReport>;

    /// Run the `solve.rs` currently installed in the workspace.
    fn run_file(&self, id: TaskId, part: u8) -> Result<RunReport> {
        let path = self.workspace(id)?.join("solve.rs");
        let source = fs::read_to_string(&path)
            .with_context(|| format!("no solution at {}", path.display()))?;
        self.run(id, part, &source)
    }

    /// Record an answer and report what is known about it.
    ///
    /// The local implementation never contacts AoC; a live implementation
    /// could submit the answer here.
    fn submit(&self, id: TaskId, part: u8, answer: &str) -> Result<Verdict>;
}

/// `Aoc` backed by the scraped archive in `.aoc/` and Rust solutions in
/// `solutions/`. No network access.
pub struct ArchiveAoc {
    archive: PathBuf,
    solutions: PathBuf,
    build_dir: PathBuf,
}

impl ArchiveAoc {
    pub fn new(archive: PathBuf, solutions: PathBuf, build_dir: PathBuf) -> Self {
        Self {
            archive,
            solutions,
            build_dir,
        }
    }

    fn day_dir(&self, id: TaskId) -> PathBuf {
        scrape::puzzle_dir(&self.archive, id.year, id.day)
    }

    fn solution_dir(&self, id: TaskId) -> PathBuf {
        self.solutions
            .join(id.year.to_string())
            .join(format!("{:02}", id.day))
    }

    fn input_path(&self, id: TaskId) -> PathBuf {
        self.day_dir(id).join("input.txt")
    }

    fn recorded(&self, id: TaskId) -> BTreeMap<String, String> {
        let path = self.day_dir(id).join("answers.json");
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn known_answer(&self, id: TaskId, part: u8) -> Option<String> {
        let recorded = self.recorded(id);
        if let Some(answer) = recorded.get(&part.to_string()) {
            return Some(answer.clone());
        }
        let puzzle = scrape::load_puzzle(&self.archive, id.year, id.day).ok()?;
        puzzle
            .answers
            .get((part - 1) as usize)
            .and_then(|answer| answer.clone())
    }

    fn write_answers(&self, id: TaskId, part: u8, answer: &str) -> Result<()> {
        let mut answers = self.recorded(id);
        answers.insert(part.to_string(), answer.to_string());
        let path = self.day_dir(id).join("answers.json");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, serde_json::to_string_pretty(&answers)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    fn install_source(&self, id: TaskId, source: &str) -> Result<PathBuf> {
        let dir = self.solution_dir(id);
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join("solve.rs");
        // Avoid touching the mtime when the code is unchanged: the build
        // cache keys off it, and the agent evaluates then runs the same file.
        if fs::read_to_string(&path).ok().as_deref() == Some(source) {
            return Ok(path);
        }
        fs::write(&path, source).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

impl Aoc for ArchiveAoc {
    fn workspace(&self, id: TaskId) -> Result<PathBuf> {
        Ok(self.solution_dir(id))
    }

    fn tasks(&self) -> Result<Vec<TaskSummary>> {
        let mut tasks = Vec::new();
        for year in scrape::available_years(&self.archive) {
            for entry in year.days.iter().filter(|entry| entry.scraped) {
                let id = TaskId::new(year.year, entry.day);
                let solved = self
                    .recorded(id)
                    .keys()
                    .filter_map(|part| part.parse().ok())
                    .collect();
                tasks.push(TaskSummary {
                    id,
                    title: entry
                        .title
                        .clone()
                        .unwrap_or_else(|| format!("Day {}", entry.day)),
                    tests: entry.tests,
                    parts: entry.parts_captured,
                    has_input: self.input_path(id).is_file(),
                    solved,
                });
            }
        }
        Ok(tasks)
    }

    fn load(&self, id: TaskId) -> Result<Task> {
        let puzzle = scrape::load_puzzle(&self.archive, id.year, id.day)?;
        let dir = self.day_dir(id);
        let statement = fs::read_to_string(dir.join("puzzle.md"))
            .with_context(|| format!("reading {}/puzzle.md", dir.display()))?;
        let input = fs::read_to_string(self.input_path(id)).ok();

        Ok(Task {
            id,
            title: puzzle.title.clone(),
            url: puzzle.url.clone(),
            statement,
            tests: tests_from_puzzle(&puzzle),
            input,
            parts: puzzle.parts.iter().map(|part| part.number).collect(),
        })
    }

    fn evaluate(&self, id: TaskId, source: &str) -> Result<TestReport> {
        let puzzle = scrape::load_puzzle(&self.archive, id.year, id.day)?;
        let source_path = self.install_source(id, source)?;
        let solution = runner::rust(&source_path, &self.build_dir, id.year, id.day)?;

        let mut cases = Vec::new();
        for test in tests_from_puzzle(&puzzle) {
            let outcome = runner::execute(&solution, id.year, id.day, test.part, &test.input)?;
            let actual = outcome.stdout.trim().to_string();
            cases.push(CaseResult {
                part: test.part,
                expected: test.expected.clone(),
                passed: outcome.success && actual == test.expected.trim(),
                actual,
                stderr: outcome.stderr,
                millis: outcome.elapsed.as_millis() as u64,
            });
        }
        Ok(TestReport { cases })
    }

    fn run(&self, id: TaskId, part: u8, source: &str) -> Result<RunReport> {
        let input = fs::read_to_string(self.input_path(id))
            .with_context(|| format!("no puzzle input at {}", self.input_path(id).display()))?;

        let source_path = self.install_source(id, source)?;
        let solution = runner::rust(&source_path, &self.build_dir, id.year, id.day)?;
        let outcome = runner::execute(&solution, id.year, id.day, part, &input)?;

        let answer = outcome.stdout.trim().to_string();
        let known = self.known_answer(id, part);
        let verdict = match known {
            Some(expected) if expected.trim() == answer => Verdict::Correct,
            Some(_) => Verdict::Incorrect,
            None => Verdict::Unverified,
        };

        Ok(RunReport {
            part,
            answer,
            stderr: outcome.stderr,
            success: outcome.success,
            millis: outcome.elapsed.as_millis() as u64,
            verdict,
        })
    }

    fn submit(&self, id: TaskId, part: u8, answer: &str) -> Result<Verdict> {
        let verdict = match self.known_answer(id, part) {
            Some(expected) if expected.trim() == answer.trim() => Verdict::Correct,
            Some(_) => Verdict::Incorrect,
            None => Verdict::Unverified,
        };
        if verdict != Verdict::Incorrect {
            self.write_answers(id, part, answer.trim())?;
        }
        Ok(verdict)
    }
}

pub fn tests_from_puzzle(puzzle: &Puzzle) -> Vec<TestCase> {
    puzzle
        .parts
        .iter()
        .flat_map(|part| {
            part.examples.iter().filter_map(|example| {
                example.expected.as_ref().map(|expected| TestCase {
                    part: part.number,
                    input: example.input.clone(),
                    expected: expected.clone(),
                })
            })
        })
        .collect()
}

pub fn input_path(archive: &Path, id: TaskId) -> PathBuf {
    scrape::puzzle_dir(archive, id.year, id.day).join("input.txt")
}
