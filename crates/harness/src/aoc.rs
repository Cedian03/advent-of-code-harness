use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tracing::warn;

use aoc_archive::model::Puzzle;
use aoc_archive::remote::RemoteDayState;
use aoc_archive::scrape;

use crate::runner;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    /// Parts whose answers are already stored locally or reported by the site.
    pub recorded: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSummary {
    pub id: TaskId,
    pub title: String,
    pub tests: u32,
    pub parts: u8,
    pub has_input: bool,
    /// Parts already answered and recorded.
    pub recorded: Vec<u8>,
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
pub enum SubmitStatus {
    Accepted,
    Rejected,
    TooSoon { wait_seconds: Option<u64> },
    WrongLevel,
    Unknown,
}

impl SubmitStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::TooSoon { .. } => "too_soon",
            Self::WrongLevel => "wrong_level",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SubmitReport {
    pub status: SubmitStatus,
    pub verdict: Verdict,
    pub message: String,
    /// Set when submitting Part One revealed Part Two.
    pub part2_unlocked: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecordReport {
    /// Whether the Answer was written to `answers.json`.
    pub written: bool,
    pub verdict: Verdict,
    pub message: String,
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

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Incorrect => "incorrect",
            Self::Unverified => "unverified",
        }
    }
}

/// One logged Submit: the Answer, the site's status, and the time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    /// Unix seconds.
    pub at: u64,
    pub part: u8,
    pub answer: String,
    /// `accepted`, `rejected`, `too_soon`, `wrong_level`, or `unknown`.
    pub status: String,
    /// `correct`, `incorrect`, `unverified`, or absent.
    pub verdict: Option<String>,
    pub message: String,
    pub wait_seconds: Option<u64>,
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

    /// Store an Answer in `answers.json`, unless it is known wrong or was
    /// rejected by the site. Recording never posts.
    fn record(&self, id: TaskId, part: u8, answer: &str) -> Result<RecordReport>;

    /// The one Part a Submit can be accepted for, derived from cached site
    /// state. `None` when both Parts are accepted or no state is known.
    fn open_part(&self, _id: TaskId) -> Result<Option<u8>> {
        Ok(None)
    }

    /// Post one Answer to adventofcode.com. Only a submit-enabled invocation
    /// exposes this through a tool; the default refuses.
    fn submit(&self, _id: TaskId, _part: u8, _answer: &str) -> Result<SubmitReport> {
        bail!("submission is disabled for this run")
    }

    /// Verdict for a candidate Answer against the local oracle.
    fn verdict(&self, _id: TaskId, _part: u8, _answer: &str) -> Verdict {
        Verdict::Unverified
    }

    /// Whether the site already refused this Answer for this Part.
    fn is_rejected(&self, _id: TaskId, _part: u8, _answer: &str) -> bool {
        false
    }

    /// Submissions logged for this Day, oldest first.
    fn submissions(&self, _id: TaskId) -> Vec<Submission> {
        Vec::new()
    }
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
        fs::read_to_string(answers_path(&self.archive, id))
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
        let path = answers_path(&self.archive, id);
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
                let recorded = self
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
                    recorded,
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
        let mut recorded: Vec<u8> = self
            .recorded(id)
            .keys()
            .filter_map(|part| part.parse().ok())
            .collect();
        for (index, answer) in puzzle.answers.iter().enumerate() {
            if answer.is_some() {
                recorded.push((index + 1) as u8);
            }
        }
        recorded.sort_unstable();
        recorded.dedup();

        Ok(Task {
            id,
            title: puzzle.title.clone(),
            url: puzzle.url.clone(),
            statement,
            tests: tests_from_puzzle(&puzzle),
            input,
            parts: puzzle.parts.iter().map(|part| part.number).collect(),
            recorded,
        })
    }

    fn evaluate(&self, id: TaskId, source: &str) -> Result<TestReport> {
        let puzzle = scrape::load_puzzle(&self.archive, id.year, id.day)?;
        let source_path = self.install_source(id, source)?;
        let solution = runner::rust(&source_path, &self.build_dir, id.year, id.day)?;

        let mut cases = Vec::new();
        for test in tests_from_puzzle(&puzzle) {
            let outcome = runner::execute(&solution, id.year, id.day, test.part, &test.input)?;
            let actual = answer_from_stdout(&outcome.stdout)
                .unwrap_or("")
                .to_string();
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

        let answer = answer_from_stdout(&outcome.stdout)
            .unwrap_or("")
            .to_string();
        let verdict = self.verdict(id, part, &answer);

        Ok(RunReport {
            part,
            answer,
            stderr: outcome.stderr,
            success: outcome.success,
            millis: outcome.elapsed.as_millis() as u64,
            verdict,
        })
    }

    fn record(&self, id: TaskId, part: u8, answer: &str) -> Result<RecordReport> {
        let answer = answer.trim();
        if answer.is_empty() {
            return Ok(RecordReport {
                written: false,
                verdict: Verdict::Unverified,
                message: "not recorded: no answer on stdout".to_string(),
            });
        }
        let verdict = self.verdict(id, part, answer);
        if verdict == Verdict::Incorrect {
            return Ok(RecordReport {
                written: false,
                verdict,
                message: "not recorded: does not match the archived answer".to_string(),
            });
        }
        if self.is_rejected(id, part, answer) {
            return Ok(RecordReport {
                written: false,
                verdict,
                message: "not recorded: answer was rejected earlier".to_string(),
            });
        }
        self.write_answers(id, part, answer)?;
        Ok(RecordReport {
            written: true,
            verdict,
            message: match verdict {
                Verdict::Correct => "recorded; matches the archived answer".to_string(),
                Verdict::Incorrect => unreachable!("handled above"),
                Verdict::Unverified => "recorded; no archived answer".to_string(),
            },
        })
    }

    fn verdict(&self, id: TaskId, part: u8, answer: &str) -> Verdict {
        match self.known_answer(id, part) {
            Some(expected) if expected.trim() == answer.trim() => Verdict::Correct,
            Some(_) => Verdict::Incorrect,
            None => Verdict::Unverified,
        }
    }

    fn is_rejected(&self, id: TaskId, part: u8, answer: &str) -> bool {
        is_rejected(&self.archive, id, part, answer)
    }

    fn submissions(&self, id: TaskId) -> Vec<Submission> {
        read_submissions(&self.archive, id)
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

/// The Answer is the last non-empty line of stdout; everything before it is
/// diagnostic output. One extractor keeps examples and real runs in agreement.
pub fn answer_from_stdout(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
}

fn answers_path(archive: &Path, id: TaskId) -> PathBuf {
    scrape::puzzle_dir(archive, id.year, id.day).join("answers.json")
}

pub fn submission_log_path(archive: &Path, id: TaskId) -> PathBuf {
    scrape::puzzle_dir(archive, id.year, id.day).join("submissions.jsonl")
}

/// Read the append-only Submission log, skipping malformed lines.
pub fn read_submissions(archive: &Path, id: TaskId) -> Vec<Submission> {
    let path = submission_log_path(archive, id);
    let Ok(text) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut submissions = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Submission>(line) {
            Ok(submission) => submissions.push(submission),
            Err(err) => warn!(
                path = %path.display(),
                line = index + 1,
                %err,
                "skipping malformed submission log line"
            ),
        }
    }
    submissions
}

pub fn append_submission(archive: &Path, id: TaskId, submission: &Submission) -> Result<()> {
    let path = submission_log_path(archive, id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(submission)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Whether the site already refused this exact Answer for this Part.
pub fn is_rejected(archive: &Path, id: TaskId, part: u8, answer: &str) -> bool {
    let answer = answer.trim();
    read_submissions(archive, id).iter().any(|entry| {
        entry.part == part && entry.status == "rejected" && entry.answer.trim() == answer
    })
}

/// Remove a refused Answer from `answers.json`; a stored answer the site
/// rejected must not remain a local oracle.
pub fn remove_recorded_answer(archive: &Path, id: TaskId, part: u8, answer: &str) -> Result<()> {
    let path = answers_path(archive, id);
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(());
    };
    let mut answers: BTreeMap<String, String> = serde_json::from_str(&text).unwrap_or_default();
    let key = part.to_string();
    let matches = answers
        .get(&key)
        .is_some_and(|stored| stored.trim() == answer.trim());
    if !matches {
        return Ok(());
    }
    answers.remove(&key);
    fs::write(&path, serde_json::to_string_pretty(&answers)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// The one Part a Submit can be accepted for.
pub fn open_part(state: &RemoteDayState) -> Option<u8> {
    if !state.part1.solved {
        Some(1)
    } else if !state.part2.solved {
        Some(2)
    } else {
        None
    }
}

/// Whether a requested Part can be acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartAvailability {
    /// The statement is in the archive.
    Available,
    /// The site has not reached this Day yet; waiting resolves it.
    Unreleased,
    /// Part Two before Part One is accepted; submitting Part One resolves it.
    Sealed,
    /// Released and unsealed, but the statement was never scraped.
    Missing,
}

impl PartAvailability {
    /// `None` when available, otherwise the message the CLI should report.
    pub fn error(self, id: TaskId, part: u8) -> Option<String> {
        match self {
            Self::Available => None,
            Self::Unreleased => Some(format!(
                "day {} of {} has not been released (unlocks {})",
                id.day,
                id.year,
                release_label(id.year, id.day)
            )),
            Self::Sealed => Some("part 2 is sealed; submit part 1 first".to_string()),
            Self::Missing => Some(format!(
                "part {part} is not in the archive; run harness scrape {id}"
            )),
        }
    }
}

/// Check availability without touching the network. The archive and the cached
/// site state are the only evidence; a Part is only "accepted" when a Star was
/// observed.
pub fn part_availability(archive: &Path, id: TaskId, part: u8) -> PartAvailability {
    let puzzle = scrape::load_puzzle(archive, id.year, id.day).ok();

    if puzzle
        .as_ref()
        .is_some_and(|puzzle| puzzle.parts.iter().any(|entry| entry.number == part))
    {
        return PartAvailability::Available;
    }

    if !is_released(id.year, id.day) {
        return PartAvailability::Unreleased;
    }

    if part == 2 {
        let cached = RemoteDayState::load(archive, id.year, id.day).map(|state| state.part1.solved);
        let derived = puzzle.as_ref().map(|puzzle| {
            RemoteDayState::from_puzzle(id.year, id.day, puzzle, 0)
                .part1
                .solved
        });
        if !cached.or(derived).unwrap_or(false) {
            return PartAvailability::Sealed;
        }
    }

    PartAvailability::Missing
}

/// A Day releases at 05:00 UTC on Dec 1 + (day − 1).
pub fn release_unix(year: u32, day: u32) -> i64 {
    (days_from_civil(year as i64, 12, 1) + (day as i64 - 1)) * 86_400 + 5 * 3_600
}

pub fn is_released(year: u32, day: u32) -> bool {
    now_unix() >= release_unix(year, day)
}

pub fn release_label(year: u32, day: u32) -> String {
    let days = days_from_civil(year as i64, 12, 1) + (day as i64 - 1);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T05:00Z")
}

pub fn current_year_month() -> (u32, u32) {
    let (year, month, _) = civil_from_days(now_unix().div_euclid(86_400));
    (year as u32, month)
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

/// Howard Hinnant's `civil_from_days`, without pulling in a date crate.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { y + 1 } else { y }, month, day)
}

/// Inverse of [`civil_from_days`].
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = (year - era * 400) as u64;
    let mp = if month > 2 { month - 3 } else { month + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + day as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_instant_is_five_utc_on_the_day() {
        assert_eq!(release_label(2025, 1), "2025-12-01T05:00Z");
        assert_eq!(release_label(2026, 2), "2026-12-02T05:00Z");
        assert_eq!(release_label(2025, 25), "2025-12-25T05:00Z");
    }

    #[test]
    fn past_days_are_released() {
        assert!(is_released(2025, 1));
        assert!(is_released(2015, 25));
    }

    #[test]
    fn answer_is_the_last_non_empty_stdout_line() {
        assert_eq!(answer_from_stdout("debug\n42\n\n"), Some("42"));
        assert_eq!(answer_from_stdout("  \n\t\n"), None);
    }

    #[test]
    fn archive_cannot_submit() {
        let dir = std::env::temp_dir().join(format!("harness-no-submit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let aoc = ArchiveAoc::new(dir.clone(), dir.join("solutions"), dir.join("target"));
        let err = aoc.submit(TaskId::new(2015, 1), 1, "42").unwrap_err();
        assert!(err.to_string().contains("submission is disabled"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn record_refuses_an_empty_answer() {
        let dir = std::env::temp_dir().join(format!("harness-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let aoc = ArchiveAoc::new(dir.clone(), dir.join("solutions"), dir.join("target"));
        let report = aoc.record(TaskId::new(2015, 1), 1, "  \n").unwrap();
        assert!(!report.written);
        assert!(report.message.contains("no answer"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejected_answers_are_remembered_and_removed() {
        let dir = std::env::temp_dir().join(format!("harness-rejected-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let id = TaskId::new(2015, 1);
        let day_dir = scrape::puzzle_dir(&dir, id.year, id.day);
        fs::create_dir_all(&day_dir).unwrap();
        fs::write(day_dir.join("answers.json"), r#"{"1":"wrong"}"#).unwrap();

        append_submission(
            &dir,
            id,
            &Submission {
                at: 1,
                part: 1,
                answer: "wrong".to_string(),
                status: "rejected".to_string(),
                verdict: Some("incorrect".to_string()),
                message: "not the right answer".to_string(),
                wait_seconds: None,
            },
        )
        .unwrap();

        assert!(is_rejected(&dir, id, 1, "wrong"));
        remove_recorded_answer(&dir, id, 1, "wrong").unwrap();
        let answers = fs::read_to_string(day_dir.join("answers.json")).unwrap();
        assert!(!answers.contains("wrong"));

        let aoc = ArchiveAoc::new(dir.clone(), dir.join("solutions"), dir.join("target"));
        let report = aoc.record(id, 1, "wrong").unwrap();
        assert!(!report.written);
        assert!(report.message.contains("rejected earlier"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn part_availability_distinguishes_unreleased_sealed_and_missing() {
        use aoc_archive::model::Part;

        let dir = std::env::temp_dir().join(format!("harness-parts-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let id = TaskId::new(2015, 1);
        let day_dir = scrape::puzzle_dir(&dir, id.year, id.day);
        fs::create_dir_all(&day_dir).unwrap();
        let puzzle = Puzzle {
            year: id.year,
            day: id.day,
            title: "Example".to_string(),
            url: String::new(),
            fetched_at: 0,
            authenticated: false,
            parts: vec![Part {
                number: 1,
                heading: String::new(),
                markdown: String::new(),
                html: String::new(),
                examples: Vec::new(),
            }],
            answers: vec![None],
            input: None,
        };
        fs::write(
            day_dir.join("puzzle.json"),
            serde_json::to_string(&puzzle).unwrap(),
        )
        .unwrap();

        assert_eq!(part_availability(&dir, id, 1), PartAvailability::Available);
        assert_eq!(
            part_availability(&dir, id, 2),
            PartAvailability::Sealed,
            "Part Two before Part One is accepted"
        );

        // A Star for Part One makes Part Two available on the site, so its
        // absence from the archive is now a capture problem, not sealing.
        let mut state = RemoteDayState::new(id.year, id.day);
        state.part1.solved = true;
        state.stars = 1;
        state.save(&dir).unwrap();
        assert_eq!(part_availability(&dir, id, 2), PartAvailability::Missing);

        assert_eq!(
            part_availability(&dir, TaskId::new(2100, 1), 1),
            PartAvailability::Unreleased
        );

        let _ = fs::remove_dir_all(dir);
    }
}
