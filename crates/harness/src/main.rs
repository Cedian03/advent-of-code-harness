use harness::address::{self, Selection};
use harness::agent::{Agent, AgentEvent, SolveReport};
use harness::aoc::{
    self, Aoc, ArchiveAoc, PartAvailability, RunReport, TaskId, TestReport, Verdict,
    answer_from_stdout, open_part, part_availability, tests_from_puzzle,
};
use harness::model::{Model, OpenAiModel, ReplayModel};
use harness::part::{self, PartSelector};
use harness::remote::{RemoteAoc, session_cookie_optional};
use harness::runner;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use aoc_archive::client::Fetcher;
use aoc_archive::remote::{RemoteDayState, calendar_stars};
use aoc_archive::scrape::{self, ScrapeOptions, load_puzzle};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "harness",
    version,
    about = "Advent of Code harness: archive puzzles, scaffold a solution, test it against examples, run it on your input"
)]
struct Cli {
    /// Local puzzle archive (gitignored; inputs are personal).
    #[arg(long, global = true, default_value = ".aoc")]
    archive: PathBuf,

    /// Directory holding per-day solutions.
    #[arg(long, global = true, default_value = "solutions")]
    solutions: PathBuf,

    /// Build directory for compiled solutions.
    #[arg(long, global = true, default_value = "target/aoc")]
    build_dir: PathBuf,

    /// Log filter, e.g. "debug" or "harness=debug,aoc_archive=trace".
    /// Also reads HARNESS_LOG, then RUST_LOG. Default: warn.
    #[arg(long, global = true)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download puzzle statements (and inputs, if AOC_SESSION is set)
    Scrape(ScrapeArgs),
    /// Import your personal puzzle input into the archive
    Input(InputArgs),
    /// Print a captured statement, or emit the task bundle as JSON
    Show(ShowArgs),
    /// Report state: Events without a Selection, Days with one
    List(ListArgs),
    /// Scaffold a solution for a day
    Init(InitArgs),
    /// Run the solution on examples or the real input
    Run(RunArgs),
    /// Drive an LLM loop that writes, tests, and runs a solution
    Solve(SolveArgs),
    /// Pick the next day with a Part that is not Recorded
    Next(NextArgs),
}

#[derive(Args)]
struct ScrapeArgs {
    /// Days or Events to scrape. Repeat addresses; let the shell expand spans:
    /// `harness scrape 20{15..25}`. No addresses means every past Event.
    #[arg(value_name = "SELECTION", value_parser = address::parse_address)]
    selection: Vec<String>,

    /// Minimum seconds between HTTP requests. AoC asks for at most one per second.
    #[arg(long, default_value_t = 1.1)]
    delay: f64,

    /// Refetch pages even when they are already cached.
    #[arg(long)]
    refresh: bool,

    /// Re-parse cached pages without refetching (after parser changes).
    #[arg(long)]
    reparse: bool,

    /// Wait for one Day to be released. The value caps the wait (default 24h).
    #[arg(long, value_name = "DUR", num_args = 0..=1, default_missing_value = "24h")]
    wait: Option<String>,

    /// Environment variable holding your adventofcode.com session cookie.
    #[arg(long, default_value = "AOC_SESSION")]
    cookie_env: String,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ListArgs {
    /// Days or Events to list. Repeat addresses; let the shell expand spans.
    #[arg(value_name = "SELECTION", value_parser = address::parse_address)]
    selection: Vec<String>,

    /// Refresh the Stars column from the site. Requires a Session.
    #[arg(long)]
    stars: bool,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ShowArgs {
    /// Puzzle to show, e.g. 2025/1.
    #[arg(value_parser = address::parse_day)]
    day: TaskId,

    /// Which part to show (default: all captured parts).
    #[arg(long, value_parser = part::parse_part, default_value = "all")]
    part: PartSelector,

    /// Emit the task bundle as JSON instead of Markdown.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct InitArgs {
    /// Puzzle to scaffold, e.g. 2025/1.
    #[arg(value_parser = address::parse_day)]
    day: TaskId,

    /// Copy a puzzle input into the archive (never committed).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Overwrite an existing solve.rs.
    #[arg(long)]
    force: bool,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct InputArgs {
    /// Puzzle to import an input for, e.g. 2025/1.
    #[arg(value_parser = address::parse_day)]
    day: TaskId,

    /// Read the input from this file instead of stdin.
    #[arg(long)]
    file: Option<PathBuf>,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct NextArgs {
    /// Event to search, e.g. 2025. Defaults to every archived Event.
    #[arg(value_parser = address::parse_event)]
    year: Option<u32>,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct SolveArgs {
    /// Puzzle to solve, e.g. 2025/1.
    #[arg(value_parser = address::parse_day)]
    day: TaskId,

    /// Which part(s) to solve: 1, 2, or all (default: all).
    #[arg(long, value_parser = part::parse_part, default_value = "all")]
    part: PartSelector,

    /// Wait for the Day to be released. The value caps the wait (default 24h).
    #[arg(long, value_name = "DUR", num_args = 0..=1, default_missing_value = "24h")]
    wait: Option<String>,

    /// Let the agent submit answers to adventofcode.com. Requires a Session.
    #[arg(long, conflicts_with = "no_submit")]
    submit: bool,

    /// Record answers locally instead of submitting them.
    #[arg(long)]
    no_submit: bool,

    /// Model id (or HARNESS_MODEL / OPENAI_MODEL).
    #[arg(long)]
    model: Option<String>,

    /// OpenAI-compatible base URL (or HARNESS_BASE_URL / OPENAI_BASE_URL).
    #[arg(long)]
    base_url: Option<String>,

    /// Replay assistant replies from a local JSON file instead of calling a model.
    #[arg(long)]
    replay: Option<PathBuf>,

    /// Maximum model turns before giving up.
    #[arg(long, default_value_t = 12)]
    turns: u32,

    /// Per-request model timeout in seconds.
    #[arg(long, default_value_t = 180)]
    timeout: u64,

    /// Cap generated tokens per reply. 0 uses the model's/provider's default.
    #[arg(long, default_value_t = 0)]
    max_tokens: u32,

    /// Suppress per-turn progress.
    #[arg(long)]
    quiet: bool,

    /// Emit structured agent events as JSON lines on stderr.
    #[arg(long)]
    events: bool,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct RunArgs {
    /// Puzzle to run, e.g. 2025/1.
    #[arg(value_parser = address::parse_day)]
    day: TaskId,

    /// Run the detected example tests instead of the real input.
    #[arg(long)]
    examples: bool,

    /// Override the solution command (sh -c; input on stdin, AOC_PART set).
    #[arg(long)]
    cmd: Option<String>,

    /// Which part(s) to run: 1, 2, or all (default: 1).
    #[arg(long, value_parser = part::parse_part, default_value = "1")]
    part: PartSelector,

    #[arg(long)]
    json: bool,
}

/// Submission is opt-in. The default lives here and nowhere else.
const SUBMIT_BY_DEFAULT: bool = false;

/// A usage or precondition error: the invocation is impossible, so exit 2.
#[derive(Debug)]
struct UsageError(String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

fn usage(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(UsageError(message.into()))
}

fn main() {
    silence_sigpipe();
    let cli = Cli::parse();
    let dotenv_result = dotenvy::dotenv();
    init_logging(cli.log_level.as_deref());
    match &dotenv_result {
        Ok(path) => {
            tracing::debug!(path = %path.display(), "loaded .env");
            warn_shadowed_env(path);
        }
        Err(err) if err.not_found() => tracing::debug!("no .env file found"),
        Err(err) => tracing::warn!(%err, "could not load .env"),
    }
    tracing::debug!(
        archive = %cli.archive.display(),
        solutions = %cli.solutions.display(),
        "harness starting"
    );
    if let Err(err) = run_cli(cli) {
        if err.downcast_ref::<UsageError>().is_some() {
            eprintln!("error: {err}");
            std::process::exit(2);
        }
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run_cli(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Scrape(args) => scrape_command(args, &cli.archive),
        Command::Input(args) => input_command(args, &cli.archive),
        Command::Show(args) => show_command(args, &cli.archive, &cli.solutions),
        Command::List(args) => list_command(args, &cli.archive, &cli.solutions),
        Command::Init(args) => init_command(args, &cli.archive, &cli.solutions),
        Command::Run(args) => run_command(args, &cli.archive, &cli.solutions, &cli.build_dir),
        Command::Solve(args) => solve_command(args, &cli.archive, &cli.solutions, &cli.build_dir),
        Command::Next(args) => next_command(args, &cli.archive, &cli.solutions, &cli.build_dir),
    }
}

/// `.env` does not overwrite variables that are already exported. Surface that,
/// because a stale `export HARNESS_BASE_URL=...` otherwise silently wins.
fn warn_shadowed_env(path: &Path) {
    let Ok(entries) = dotenvy::from_path_iter(path) else {
        return;
    };
    for (key, value) in entries.flatten() {
        if let Ok(existing) = std::env::var(&key)
            && existing != value
        {
            tracing::warn!(
                variable = %key,
                "environment variable overrides .env; unset it or update the shell export"
            );
        }
    }
}

fn init_logging(level: Option<&str>) {
    use tracing_subscriber::EnvFilter;

    let filter = level
        .map(EnvFilter::new)
        .or_else(|| EnvFilter::try_from_env("HARNESS_LOG").ok())
        .or_else(|| EnvFilter::try_from_default_env().ok())
        .unwrap_or_else(|| EnvFilter::new("warn"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Rust ignores SIGPIPE, which turns `harness list | head` into a panic on a
/// closed stdout. Restore the usual Unix behavior of exiting quietly.
#[cfg(unix)]
fn silence_sigpipe() {
    // SAFETY: setting a signal disposition to the default is always valid.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn silence_sigpipe() {}

fn scrape_command(args: ScrapeArgs, archive: &Path) -> Result<()> {
    if args.delay < 1.0 {
        return Err(usage(
            "--delay must be at least 1.0 seconds; AoC asks clients to make at most one request per second",
        ));
    }

    let selection = Selection::parse(&args.selection)?;
    if let Some(value) = &args.wait {
        let Some(id) = selection.single_day() else {
            return Err(usage(
                "--wait requires exactly one Day, e.g. `harness scrape 2025/1 --wait`",
            ));
        };
        wait_for_day(id, parse_duration(value)?)?;
    }

    let (years, days) = scrape_targets(&selection)?;
    let summary_days = days.clone();

    let cookie = std::env::var(&args.cookie_env)
        .ok()
        .filter(|value| !value.trim().is_empty());
    if cookie.is_none() {
        warning("no AOC_SESSION: public pages only — no inputs, no Part Two, no stars");
    }

    let root = scrape::run(&ScrapeOptions {
        years,
        days,
        out_dir: archive.to_path_buf(),
        delay: Duration::from_secs_f64(args.delay),
        refresh: args.refresh,
        reparse: args.reparse,
        cookie,
    })?;

    let entries: Vec<_> = root
        .years
        .iter()
        .flat_map(|year| year.days.iter().map(move |day| (year.year, day)))
        .filter(|(year, day)| {
            summary_days
                .as_ref()
                .is_none_or(|days| days.contains(&(*year, day.day)))
        })
        .collect();

    if args.json {
        let days: Vec<_> = entries
            .iter()
            .map(|(year, day)| {
                serde_json::json!({
                    "year": year,
                    "day": day.day,
                    "address": TaskId::new(*year, day.day).to_string(),
                    "title": day.title,
                    "captured": day.scraped,
                    "parts_captured": day.parts_captured,
                    "tests": day.tests,
                    "input": day.input,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "days": days }))?
        );
        return Ok(());
    }

    let total = entries.len();
    let part2 = entries
        .iter()
        .filter(|(_, day)| day.parts_captured >= 2)
        .count();
    let inputs = entries
        .iter()
        .filter(|(_, day)| day.input.is_some())
        .count();
    let years = entries
        .iter()
        .map(|(year, _)| *year)
        .collect::<BTreeSet<_>>()
        .len();
    eprintln!(
        "\nArchive: {total} day(s) across {years} year(s); {part2} with Part Two; {inputs} with inputs."
    );
    Ok(())
}

/// Years plus an optional exact Day set, as the scraper expects.
type ScrapeTargets = (Vec<u32>, Option<BTreeSet<(u32, u32)>>);

/// Resolve a Selection into the scraper's years plus exact Days. An empty
/// Selection means every past Event and all of its Days.
fn scrape_targets(selection: &Selection) -> Result<ScrapeTargets> {
    match selection.day_set() {
        None => Ok((default_years(), None)),
        Some(days) => {
            let years = days
                .iter()
                .map(|(year, _)| *year)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            Ok((years, Some(days)))
        }
    }
}

fn warning(message: &str) {
    eprintln!("warning: {message}");
}

/// Parse a `--wait` ceiling such as `30s`, `90m`, or `24h`.
fn parse_duration(value: &str) -> Result<Duration> {
    let value = value.trim();
    let (number, unit) = value.split_at(value.len().saturating_sub(1));
    let amount: u64 = number
        .parse()
        .map_err(|_| usage(format!("invalid duration `{value}`; use e.g. 90m or 24h")))?;
    let seconds = match unit {
        "s" => amount,
        "m" => amount.saturating_mul(60),
        "h" => amount.saturating_mul(3_600),
        _ => return Err(usage(format!("invalid duration `{value}`; use s, m, or h"))),
    };
    Ok(Duration::from_secs(seconds))
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3_600 {
        format!("{}h{}m", seconds / 3_600, (seconds % 3_600) / 60)
    } else if seconds >= 60 {
        format!("{}m{}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

/// Hold until a Day is released, refusing when the release is beyond the
/// ceiling. An already released Day is a no-op.
fn wait_for_day(id: TaskId, ceiling: Duration) -> Result<()> {
    let release = aoc::release_unix(id.year, id.day);
    let now = aoc::now_unix();
    if now >= release {
        return Ok(());
    }

    let remaining = Duration::from_secs((release - now) as u64);
    let instant = aoc::release_label(id.year, id.day);
    if remaining > ceiling {
        return Err(usage(format!(
            "{id} has not been released (unlocks {instant}); waiting {} exceeds the {} ceiling",
            format_duration(remaining),
            format_duration(ceiling)
        )));
    }

    eprintln!(
        "waiting for {id}: releases {instant} ({})",
        format_duration(remaining)
    );
    let lead = remaining.saturating_sub(Duration::from_secs(120));
    if !lead.is_zero() {
        std::thread::sleep(lead);
    }

    loop {
        let now = aoc::now_unix();
        if now >= release {
            eprintln!("{id} is released");
            return Ok(());
        }
        let remaining = (release - now) as u64;
        let jitter = (now.unsigned_abs() % 4) as u64;
        eprintln!(
            "waiting for {id}: {} remaining",
            format_duration(Duration::from_secs(remaining))
        );
        std::thread::sleep(Duration::from_secs((remaining.min(10) + jitter).max(1)));
    }
}

fn show_command(args: ShowArgs, archive: &Path, solutions: &Path) -> Result<()> {
    let id = args.day;
    match args.part.explicit() {
        Some(part) => ensure_part(archive, id, part)?,
        // `all` still needs Part One; an Unreleased or uncaptured Day should
        // fail with the same domain message as an explicit selector.
        None => ensure_part(archive, id, 1)?,
    }

    if args.json {
        print_task_bundle(id, archive, solutions, args.part)?;
        return Ok(());
    }

    let path = scrape::puzzle_dir(archive, id.year, id.day).join("puzzle.md");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("no captured puzzle at {}", path.display()))?;
    match args.part {
        PartSelector::All => print!("{text}"),
        PartSelector::One => print!("{}", statement_slice(&text, 1)),
        PartSelector::Two => print!("{}", statement_slice(&text, 2)),
    }
    Ok(())
}

/// Slice the generated Markdown at its `## Part Two` heading.
fn statement_slice(markdown: &str, part: u8) -> String {
    const MARKER: &str = "\n## Part Two";
    match part {
        1 => markdown
            .split(MARKER)
            .next()
            .unwrap_or(markdown)
            .trim_end()
            .to_string(),
        2 => markdown
            .find(MARKER)
            .map(|index| markdown[index..].trim_start().to_string())
            .unwrap_or_default(),
        _ => markdown.to_string(),
    }
}

/// Reject an explicitly selected Part that cannot be acted on.
fn ensure_part(archive: &Path, id: TaskId, part: u8) -> Result<()> {
    if let Some(message) = part_availability(archive, id, part).error(id, part) {
        return Err(usage(message));
    }
    Ok(())
}

fn list_command(args: ListArgs, archive: &Path, solutions: &Path) -> Result<()> {
    let selection = Selection::parse(&args.selection)?;

    if args.stars {
        let cookie = session_cookie_optional().ok_or_else(|| {
            usage("list --stars needs AOC_SESSION; set it in .env, or drop --stars")
        })?;
        let years = if selection.is_empty() {
            scrape::available_years(archive)
                .iter()
                .map(|year| year.year)
                .collect::<Vec<_>>()
        } else {
            selection.years()
        };
        if years.is_empty() {
            return Err(usage(format!(
                "no archive in {} yet — run `harness scrape` first",
                archive.display()
            )));
        }
        refresh_stars(archive, &years, cookie, Duration::from_secs_f64(1.1))?;
    }

    if selection.is_empty() {
        list_events(archive, solutions, args.json)
    } else {
        list_days(&selection.days(), archive, solutions, args.json)
    }
}

/// Refresh the cached Stars column from the site's calendar.
fn refresh_stars(archive: &Path, years: &[u32], cookie: String, delay: Duration) -> Result<()> {
    let mut fetcher = Fetcher::new(archive.to_path_buf(), Some(cookie), delay, true);
    for &year in years {
        let calendar_html = fetcher.calendar_fresh(year)?;
        let stars = calendar_stars(year, &calendar_html);
        for day in 1..=25 {
            let count = stars.get(&day).copied().unwrap_or(0);
            let existing = RemoteDayState::load(archive, year, day);
            if count == 0 && existing.is_none() {
                continue;
            }
            let mut state = existing.unwrap_or_else(|| RemoteDayState::new(year, day));
            state.stars = state.stars.max(count);
            state.part1.solved |= count >= 1;
            state.part2.solved |= count >= 2;
            if state.part1.solved {
                state.part2_unlocked = true;
            }
            state.save(archive)?;
        }
    }
    Ok(())
}

/// No Selection: one row per Event.
fn list_events(archive: &Path, _solutions: &Path, json: bool) -> Result<()> {
    struct Row {
        year: u32,
        days: usize,
        captured: usize,
        part_two: usize,
        inputs: usize,
        tests: u32,
        recorded: usize,
        stars: u32,
    }

    let years = scrape::available_years(archive);
    if years.is_empty() {
        bail!(
            "no archive in {} yet — run `harness scrape` first",
            archive.display()
        );
    }

    let rows: Vec<Row> = years
        .iter()
        .map(|year| Row {
            year: year.year,
            days: year.days.len(),
            captured: year.scraped_days(),
            part_two: year
                .days
                .iter()
                .filter(|day| day.parts_captured >= 2)
                .count(),
            inputs: year
                .days
                .iter()
                .filter(|day| has_input(archive, year.year, day.day))
                .count(),
            tests: year.days.iter().map(|day| day.tests).sum(),
            recorded: year
                .days
                .iter()
                .filter(|day| !recorded_parts(archive, TaskId::new(year.year, day.day)).is_empty())
                .count(),
            stars: year
                .days
                .iter()
                .map(|day| recorded_stars(archive, TaskId::new(year.year, day.day)) as u32)
                .sum(),
        })
        .collect();

    if json {
        let rows: Vec<_> = rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "year": row.year,
                    "days": row.days,
                    "captured": row.captured,
                    "part_two": row.part_two,
                    "inputs": row.inputs,
                    "tests": row.tests,
                    "recorded": row.recorded,
                    "stars": row.stars,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "years": rows }))?
        );
        return Ok(());
    }

    println!(
        "{:<6} {:>5} {:>9} {:>8} {:>7} {:>6} {:>9} {:>6}",
        "Year", "Days", "Captured", "Part 2", "Inputs", "Tests", "Recorded", "Stars"
    );
    for row in &rows {
        println!(
            "{:<6} {:>5} {:>9} {:>8} {:>7} {:>6} {:>9} {:>6}",
            row.year,
            row.days,
            row.captured,
            row.part_two,
            row.inputs,
            row.tests,
            row.recorded,
            row.stars
        );
    }
    Ok(())
}

/// A Selection: one row per Day, ascending.
fn list_days(days: &[TaskId], archive: &Path, solutions: &Path, json: bool) -> Result<()> {
    struct Row {
        id: TaskId,
        title: String,
        tests: usize,
        part_two: bool,
        input: bool,
        recorded: Vec<u8>,
        stars: u8,
        solution: bool,
    }

    let years = scrape::available_years(archive);
    let rows: Vec<Row> = days
        .iter()
        .map(|id| {
            let entry = years
                .iter()
                .find(|index| index.year == id.year)
                .and_then(|index| index.days.iter().find(|entry| entry.day == id.day));
            let puzzle = load_puzzle(archive, id.year, id.day).ok();
            Row {
                id: *id,
                title: puzzle
                    .as_ref()
                    .map(|puzzle| puzzle.title.clone())
                    .or_else(|| entry.and_then(|entry| entry.title.clone()))
                    .unwrap_or_else(|| "-".to_string()),
                tests: puzzle
                    .as_ref()
                    .map(|puzzle| tests_from_puzzle(puzzle).len())
                    .or_else(|| entry.map(|entry| entry.tests as usize))
                    .unwrap_or(0),
                part_two: puzzle.as_ref().is_some_and(|puzzle| puzzle.is_complete())
                    || entry.is_some_and(|entry| entry.parts_captured >= 2),
                input: has_input(archive, id.year, id.day),
                recorded: recorded_parts(archive, *id),
                stars: recorded_stars(archive, *id),
                solution: solution_exists(solutions, id.year, id.day),
            }
        })
        .collect();

    if json {
        let rows: Vec<_> = rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "year": row.id.year,
                    "day": row.id.day,
                    "address": row.id.to_string(),
                    "title": row.title,
                    "tests": row.tests,
                    "part_two_captured": row.part_two,
                    "input": row.input,
                    "recorded": row.recorded,
                    "stars": row.stars,
                    "solution": row.solution,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    println!(
        "{:<8} {:<44} {:>5} {:>3} {:>6} {:>9} {:>6} {:>9}",
        "Day", "Title", "Tests", "P2", "Input", "Recorded", "Stars", "Solution"
    );
    for row in &rows {
        println!(
            "{:<8} {:<44} {:>5} {:>3} {:>6} {:>9} {:>6} {:>9}",
            row.id,
            truncate(&row.title, 44),
            row.tests,
            yes_no(row.part_two),
            yes_no(row.input),
            row.recorded.len(),
            row.stars,
            yes_no(row.solution),
        );
    }
    Ok(())
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn recorded_parts(archive: &Path, id: TaskId) -> Vec<u8> {
    let answers_path = scrape::puzzle_dir(archive, id.year, id.day).join("answers.json");
    let mut parts: Vec<u8> = fs::read_to_string(answers_path)
        .ok()
        .and_then(|text| serde_json::from_str::<BTreeMap<String, String>>(&text).ok())
        .map(|answers| {
            answers
                .keys()
                .filter_map(|part| part.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    if let Ok(puzzle) = load_puzzle(archive, id.year, id.day) {
        for (index, answer) in puzzle.answers.iter().enumerate() {
            if answer.is_some() {
                parts.push((index + 1) as u8);
            }
        }
    }
    parts.sort_unstable();
    parts.dedup();
    parts
}

fn recorded_stars(archive: &Path, id: TaskId) -> u8 {
    RemoteDayState::load(archive, id.year, id.day)
        .map(|state| state.stars)
        .unwrap_or(0)
}

fn init_command(args: InitArgs, archive: &Path, solutions: &Path) -> Result<()> {
    let id = args.day;
    let dir = solutions
        .join(id.year.to_string())
        .join(format!("{:02}", id.day));
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let existed = ["solve", "solve.py", "solve.rs"]
        .iter()
        .any(|name| dir.join(name).is_file());
    let source = dir.join("solve.rs");
    let created = args.force || !existed;
    if created {
        fs::write(&source, RUST_TEMPLATE)
            .with_context(|| format!("writing {}", source.display()))?;
    }

    let input_path = if let Some(input) = &args.input {
        if fs::metadata(input)
            .with_context(|| format!("reading {}", input.display()))?
            .len()
            == 0
        {
            return Err(usage("input is empty; nothing imported"));
        }
        let target = scrape::puzzle_dir(archive, id.year, id.day).join("input.txt");
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(input, &target)
            .with_context(|| format!("copying {} to {}", input.display(), target.display()))?;
        Some(target)
    } else {
        None
    };

    let captured = scrape::puzzle_dir(archive, id.year, id.day)
        .join("puzzle.json")
        .is_file();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "year": id.year,
                "day": id.day,
                "address": id.to_string(),
                "solution": source,
                "created": created,
                "input": input_path,
                "captured": captured,
            }))?
        );
        return Ok(());
    }

    if created {
        println!("created {}", source.display());
    } else {
        println!("solution already exists in {}", dir.display());
    }
    if let Some(target) = &input_path {
        println!("imported input -> {}", target.display());
    }
    if !captured {
        println!("note: no archived statement for {id} — run `harness scrape {id}`");
    }
    println!("next: harness run {id} --examples");
    Ok(())
}

fn input_command(args: InputArgs, archive: &Path) -> Result<()> {
    let id = args.day;
    let bytes = match &args.file {
        Some(path) => fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        None => {
            use std::io::{IsTerminal, Read};
            if std::io::stdin().is_terminal() {
                eprintln!("paste the puzzle input, then press Ctrl-D:");
            }
            let mut bytes = Vec::new();
            std::io::stdin()
                .read_to_end(&mut bytes)
                .context("reading stdin")?;
            bytes
        }
    };

    if bytes.is_empty() {
        return Err(usage("input is empty; nothing imported"));
    }

    let path = write_input(archive, id, &bytes)?;
    let captured = scrape::puzzle_dir(archive, id.year, id.day)
        .join("puzzle.json")
        .is_file();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "year": id.year,
                "day": id.day,
                "address": id.to_string(),
                "path": path,
                "bytes": bytes.len(),
                "captured": captured,
            }))?
        );
        return Ok(());
    }

    println!("imported {} bytes -> {}", bytes.len(), path.display());
    if !captured {
        println!("note: no archived puzzle for {id} yet — run `harness scrape {id}`");
    }
    Ok(())
}

fn write_input(archive: &Path, id: TaskId, bytes: &[u8]) -> Result<PathBuf> {
    let path = aoc::input_path(archive, id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

fn print_task_bundle(
    id: TaskId,
    archive: &Path,
    solutions: &Path,
    part: PartSelector,
) -> Result<()> {
    let puzzle = load_puzzle(archive, id.year, id.day)?;
    let dir = scrape::puzzle_dir(archive, id.year, id.day);
    let full_statement = fs::read_to_string(dir.join("puzzle.md")).unwrap_or_default();
    let statement = match part.explicit() {
        Some(part) => statement_slice(&full_statement, part),
        None => full_statement,
    };
    let input = find_input(archive, id);
    let tests: Vec<_> = tests_from_puzzle(&puzzle)
        .into_iter()
        .filter(|test| part.includes(test.part))
        .collect();

    let bundle = serde_json::json!({
        "year": id.year,
        "day": id.day,
        "title": puzzle.title,
        "url": puzzle.url,
        "part_two_captured": puzzle.is_complete(),
        "parts": puzzle.parts.iter().map(|part| part.number).collect::<Vec<_>>(),
        "statement": statement,
        "input_path": input.as_ref().map(|path| path.to_string_lossy()),
        "solution_dir": solutions.join(id.year.to_string()).join(format!("{:02}", id.day)),
        "tests": tests,
    });
    println!("{}", serde_json::to_string_pretty(&bundle)?);
    Ok(())
}

fn run_examples(args: &RunArgs, archive: &Path, solutions: &Path, build_dir: &Path) -> Result<()> {
    let id = args.day;
    match args.part.explicit() {
        Some(part) => ensure_part(archive, id, part)?,
        None => ensure_part(archive, id, 1)?,
    }
    let puzzle = load_puzzle(archive, id.year, id.day)?;
    let mut tests = tests_from_puzzle(&puzzle);
    tests.retain(|test| args.part.includes(test.part));
    if tests.is_empty() {
        println!(
            "{} has no detected example tests{}.",
            id,
            if puzzle.is_complete() {
                ""
            } else {
                " (Part Two is not in the archive)"
            }
        );
        return Ok(());
    }

    let solution = runner::discover(solutions, build_dir, id.year, id.day, args.cmd.as_deref())?;

    let mut results = Vec::new();
    let mut failures = 0usize;
    for test in &tests {
        let outcome = runner::execute(&solution, id.year, id.day, test.part, &test.input)?;
        let actual = answer_from_stdout(&outcome.stdout)
            .unwrap_or("")
            .to_string();
        let expected = test.expected.trim().to_string();
        let passed = outcome.success && actual == expected;
        if !passed {
            failures += 1;
        }
        results.push((test, actual, outcome, passed));
    }

    if args.json {
        let payload: Vec<_> = results
            .iter()
            .map(|(test, actual, outcome, passed)| {
                serde_json::json!({
                    "part": test.part,
                    "expected": test.expected.trim(),
                    "actual": actual,
                    "passed": passed,
                    "stderr": outcome.stderr.trim(),
                    "millis": outcome.elapsed.as_millis() as u64,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("{} {} — {} test(s)", id, puzzle.title, tests.len());
        println!();
        for (test, actual, outcome, passed) in &results {
            if *passed {
                println!("  part {}  ok  {}", test.part, actual);
            } else {
                println!(
                    "  part {}  FAIL  expected {}, got {}",
                    test.part,
                    test.expected.trim(),
                    if actual.is_empty() {
                        "<no output>"
                    } else {
                        actual
                    }
                );
                if !outcome.stderr.trim().is_empty() {
                    for line in outcome.stderr.trim().lines().take(6) {
                        println!("               {line}");
                    }
                }
            }
        }
        println!();
        if failures == 0 {
            println!("all {} test(s) passed", tests.len());
        } else {
            println!("{failures}/{} test(s) failed", tests.len());
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn run_command(args: RunArgs, archive: &Path, solutions: &Path, build_dir: &Path) -> Result<()> {
    if args.examples {
        return run_examples(&args, archive, solutions, build_dir);
    }

    let id = args.day;
    match args.part.explicit() {
        Some(part) => ensure_part(archive, id, part)?,
        None => ensure_part(archive, id, 1)?,
    }
    let puzzle = load_puzzle(archive, id.year, id.day)?;
    let parts = run_parts(args.part, archive, id, &puzzle)?;

    let input_path = match find_input(archive, id) {
        Some(path) => path,
        None if session_cookie_optional().is_none() => {
            return Err(usage(format!(
                "no input for {id} and no AOC_SESSION to fetch it; run `harness input {id} --file <path>`"
            )));
        }
        None => bail!(
            "no puzzle input for {id}; run `harness scrape {id}` with AOC_SESSION or `harness input {id} --file <path>`"
        ),
    };
    let input = fs::read_to_string(&input_path)
        .with_context(|| format!("reading {}", input_path.display()))?;

    let solution = runner::discover(solutions, build_dir, id.year, id.day, args.cmd.as_deref())?;

    let mut results = Vec::new();
    let mut any_failed = false;
    for part in parts {
        let outcome = runner::execute(&solution, id.year, id.day, part, &input)?;
        let answer = answer_from_stdout(&outcome.stdout)
            .unwrap_or("")
            .to_string();
        let recorded = recorded_answer(archive, &puzzle, part);
        let matches = recorded
            .as_deref()
            .map(|expected| expected.trim() == answer);
        if !outcome.success {
            any_failed = true;
        }
        results.push(PartRun {
            part,
            answer,
            recorded,
            matches,
            success: outcome.success,
            stderr: outcome.stderr,
            millis: outcome.elapsed.as_millis() as u64,
        });
    }

    if args.json {
        let payload: Vec<_> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "part": result.part,
                    "answer": result.answer,
                    "recorded_answer": result.recorded,
                    "matches_recorded": result.matches,
                    "success": result.success,
                    "stderr": result.stderr.trim(),
                    "millis": result.millis,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "year": id.year,
                "day": id.day,
                "parts": payload,
            }))?
        );
    } else {
        for result in &results {
            if result.success {
                println!(
                    "{id} part {}: {}  ({} ms)",
                    result.part,
                    if result.answer.is_empty() {
                        "<no output>"
                    } else {
                        &result.answer
                    },
                    result.millis
                );
            } else {
                println!(
                    "{id} part {} failed after {} ms",
                    result.part, result.millis
                );
            }
            if !result.stderr.trim().is_empty() {
                for line in result.stderr.trim().lines().take(10) {
                    eprintln!("  {line}");
                }
            }
            match result.matches {
                Some(true) => println!("  matches the recorded answer"),
                Some(false) => println!(
                    "  does not match the recorded answer ({})",
                    result.recorded.as_deref().unwrap_or("")
                ),
                None => {}
            }
        }
    }

    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}

struct PartRun {
    part: u8,
    answer: String,
    recorded: Option<String>,
    matches: Option<bool>,
    success: bool,
    stderr: String,
    millis: u64,
}

/// Resolve `--part` to concrete Parts. An explicit Part must be available; `all`
/// means every captured Part.
fn run_parts(
    selector: PartSelector,
    archive: &Path,
    id: TaskId,
    puzzle: &aoc_archive::model::Puzzle,
) -> Result<Vec<u8>> {
    match selector.explicit() {
        Some(part) => {
            ensure_part(archive, id, part)?;
            Ok(vec![part])
        }
        None => {
            let parts: Vec<u8> = puzzle
                .parts
                .iter()
                .map(|part| part.number)
                .filter(|part| selector.includes(*part))
                .collect();
            if parts.is_empty() {
                bail!("no Part of {id} is in the archive; run `harness scrape {id}`");
            }
            Ok(parts)
        }
    }
}

fn recorded_answer(
    archive: &Path,
    puzzle: &aoc_archive::model::Puzzle,
    part: u8,
) -> Option<String> {
    let path = scrape::puzzle_dir(archive, puzzle.year, puzzle.day).join("answers.json");
    if let Ok(text) = fs::read_to_string(path)
        && let Ok(answers) = serde_json::from_str::<BTreeMap<String, String>>(&text)
        && let Some(answer) = answers.get(&part.to_string())
    {
        return Some(answer.clone());
    }
    puzzle
        .answers
        .get((part - 1) as usize)
        .and_then(|answer| answer.clone())
}

fn recorded_answer_for(archive: &Path, id: TaskId, part: u8) -> Option<String> {
    let puzzle = load_puzzle(archive, id.year, id.day).ok()?;
    recorded_answer(archive, &puzzle, part)
}

fn next_command(args: NextArgs, archive: &Path, solutions: &Path, build_dir: &Path) -> Result<()> {
    if !has_remote_state(archive) {
        warning("no cached site state: cannot tell a Recorded Day from one already Solved");
    }

    let local = ArchiveAoc::new(
        archive.to_path_buf(),
        solutions.to_path_buf(),
        build_dir.to_path_buf(),
    );
    let tasks = local.tasks()?;
    let next = tasks
        .iter()
        .filter(|task| args.year.is_none_or(|year| year == task.id.year))
        .filter(|task| task.recorded.len() < task.parts as usize)
        .min_by_key(|task| (!task.has_input, task.id.year, task.id.day));

    let Some(task) = next else {
        bail!(
            "every captured Part is Recorded{}",
            args.year
                .map(|year| format!(" for {year}"))
                .unwrap_or_default()
        );
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(task)?);
    } else {
        println!(
            "next: {} — {} ({} example test(s), {})",
            task.id,
            task.title,
            task.tests,
            if task.has_input {
                "input available"
            } else {
                "no input"
            }
        );
        println!("run: harness solve {}", task.id);
    }
    Ok(())
}

fn has_remote_state(archive: &Path) -> bool {
    scrape::available_years(archive).iter().any(|year| {
        year.days
            .iter()
            .any(|day| RemoteDayState::load(archive, year.year, day.day).is_some())
    })
}

fn build_model(args: &SolveArgs) -> Result<OpenAiModel> {
    let mut model = OpenAiModel::from_env(args.model.as_deref(), args.base_url.as_deref())?
        .with_timeout(Duration::from_secs(args.timeout));
    if args.max_tokens > 0 {
        model = model.with_max_tokens(args.max_tokens);
    }
    if !args.quiet {
        eprintln!(
            "model {} via {} (timeout {}s, max tokens: {})",
            model.name(),
            model.base_url(),
            args.timeout,
            if args.max_tokens == 0 {
                "provider default".to_string()
            } else {
                args.max_tokens.to_string()
            }
        );
    }
    Ok(model)
}

fn run_agent<A: Aoc, M: Model>(
    aoc: &A,
    id: TaskId,
    args: &SolveArgs,
    model: &M,
    submit: bool,
    emit: &mut dyn FnMut(&AgentEvent),
) -> Result<SolveReport> {
    let agent = Agent::new(aoc, model, args.turns);
    let agent = if args.quiet { agent.quiet() } else { agent };
    let agent = if submit { agent.with_submit() } else { agent };
    agent.solve_parts(id, args.part, emit)
}

fn solve_command(
    args: SolveArgs,
    archive: &Path,
    solutions: &Path,
    build_dir: &Path,
) -> Result<()> {
    if args.turns == 0 {
        return Err(usage("--turns must be at least 1"));
    }
    if args.timeout == 0 {
        return Err(usage("--timeout must be at least 1 second"));
    }

    let id = args.day;
    let cookie = session_cookie_optional();
    let replay = args.replay.as_deref();
    let submit = if args.submit {
        true
    } else if args.no_submit {
        false
    } else {
        SUBMIT_BY_DEFAULT
    } && replay.is_none();

    if submit && cookie.is_none() {
        return Err(usage(
            "--submit needs AOC_SESSION; set it in .env, or drop --submit to record only",
        ));
    }

    if let Some(value) = &args.wait {
        wait_for_day(id, parse_duration(value)?)?;
    }

    // A Day the site has not reached can never be fetched or solved.
    let mut part1 = part_availability(archive, id, 1);
    if part1 == PartAvailability::Unreleased {
        return Err(usage(
            part1
                .error(id, 1)
                .unwrap_or_else(|| "day is not available".into()),
        ));
    }

    // A replay run never fetches, so its Day must already be Captured.
    if replay.is_some() && part1 != PartAvailability::Available {
        return Err(usage(format!(
            "{}; a replay run never fetches, so scrape it first",
            part1
                .error(id, 1)
                .unwrap_or_else(|| "day is not available".into())
        )));
    }

    if cookie.is_none()
        && args.part.includes(2)
        && part_availability(archive, id, 2) != PartAvailability::Available
    {
        warning("no AOC_SESSION: Part Two cannot be unsealed; this run stops after Part One");
    }

    let remote = RemoteAoc::new(
        archive.to_path_buf(),
        solutions.to_path_buf(),
        build_dir.to_path_buf(),
        cookie.clone(),
        Duration::from_secs_f64(1.1),
    );

    if replay.is_none() {
        remote.ensure_day(id)?;
        part1 = part_availability(archive, id, 1);
    }

    if !submit && args.part.explicit() == Some(2) {
        ensure_part(archive, id, 2)?;
    }
    if part1 != PartAvailability::Available {
        let message = part1
            .error(id, 1)
            .unwrap_or_else(|| "day is not available".into());
        return Err(usage(message));
    }
    if find_input(archive, id).is_none() {
        if cookie.is_none() {
            return Err(usage(format!(
                "no input for {id} and no AOC_SESSION to fetch it; run `harness input {id} --file <path>`"
            )));
        }
        return Err(usage(format!(
            "no input for {id}; run `harness input {id} --file <path>`"
        )));
    }

    // A submit-enabled run reads the calendar once before turn one, so the
    // Open Part reflects a Part solved outside the harness. The site is never
    // asked again during a tool call.
    let mut remote_state = remote.state(id)?;
    if submit {
        let (state, _) = remote
            .verify_star(id)
            .with_context(|| format!("reading the calendar for {id}"))?;
        remote_state = state;

        // If Part Two is the Open Part and selected but its statement was
        // never captured, fetch it before the model can work.
        if args.part.includes(2) && open_part(&remote_state) == Some(2) {
            let captured = load_puzzle(archive, id.year, id.day)
                .map(|puzzle| puzzle.parts.iter().any(|part| part.number == 2))
                .unwrap_or(false);
            if !captured {
                remote_state = remote.ensure_day(id)?;
            }
        }
    }
    let open = if submit {
        open_part(&remote_state)
    } else {
        None
    };

    // `--part 2` is impossible until the site accepts Part One.
    if submit && args.part.explicit() == Some(2) && open == Some(1) {
        return Err(usage("part 2 is sealed; submit part 1 first"));
    }
    if submit
        && let Some(part) = open
        && args.part.includes(part)
    {
        ensure_part(archive, id, part)?;
    }

    let puzzle = load_puzzle(archive, id.year, id.day)?;
    let captured: Vec<u8> = puzzle
        .parts
        .iter()
        .map(|part| part.number)
        .filter(|part| args.part.includes(*part))
        .collect();
    let selected: Vec<u8> = (1u8..=2).filter(|part| args.part.includes(*part)).collect();

    let (target_parts, needs_model) = if submit {
        match open {
            Some(part) if args.part.includes(part) => (vec![part], true),
            _ => (selected.clone(), false),
        }
    } else {
        let recorded = recorded_parts(archive, id);
        let needs_model = captured.iter().any(|part| !recorded.contains(part));
        (captured, needs_model)
    };

    let report = if needs_model {
        let mut emit = |event: &AgentEvent| {
            if args.events {
                match serde_json::to_string(event) {
                    Ok(line) => eprintln!("{line}"),
                    Err(err) => tracing::warn!(%err, "could not serialize agent event"),
                }
            }
        };

        let local = ArchiveAoc::new(
            archive.to_path_buf(),
            solutions.to_path_buf(),
            build_dir.to_path_buf(),
        );
        if let Some(path) = replay {
            let model = ReplayModel::from_file(path)?;
            if submit {
                run_agent(&remote, id, &args, &model, submit, &mut emit)?
            } else {
                run_agent(&local, id, &args, &model, submit, &mut emit)?
            }
        } else {
            let model = build_model(&args)?;
            if submit {
                run_agent(&remote, id, &args, &model, submit, &mut emit)?
            } else {
                run_agent(&local, id, &args, &model, submit, &mut emit)?
            }
        }
    } else {
        // Nothing left to do: no model turn is needed.
        let runs = target_parts
            .iter()
            .filter_map(|&part| {
                recorded_answer_for(archive, id, part).map(|answer| RunReport {
                    part,
                    answer,
                    stderr: String::new(),
                    success: true,
                    millis: 0,
                    verdict: Verdict::Unverified,
                })
            })
            .collect();
        let note = if !submit && args.part.includes(2) && !target_parts.contains(&2) {
            Some("part 2 is sealed; re-run with --submit".to_string())
        } else {
            None
        };
        let recorded = recorded_parts(archive, id)
            .into_iter()
            .filter(|part| target_parts.contains(part))
            .collect();
        SolveReport {
            id,
            turns: 0,
            verified: false,
            tests: TestReport { cases: Vec::new() },
            runs,
            recorded,
            submissions: Vec::new(),
            history: Vec::new(),
            note,
        }
    };

    let star = if submit {
        match remote.verify_star(id) {
            Ok(state) => Some(state),
            Err(err) => {
                warning(&format!(
                    "could not verify the star on the calendar: {err:#}"
                ));
                None
            }
        }
    } else {
        None
    };

    let remote_state = remote.state(id).ok();
    let solved = star.as_ref().is_some_and(|(_, confirmed)| *confirmed);
    // Exit 0 with --submit only when the site accepted every selected Part;
    // the calendar may still lag behind the acceptance.
    let accepted = submit
        && selected.iter().all(|part| match part {
            1 => remote_state
                .as_ref()
                .is_some_and(|state| state.part1.solved),
            2 => remote_state
                .as_ref()
                .is_some_and(|state| state.part2.solved),
            _ => false,
        });

    if args.json {
        let payload = serde_json::json!({
            "report": &report,
            "remote": &remote_state,
            "star_confirmed": star.as_ref().map(|(_, confirmed)| *confirmed),
            "accepted": accepted,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        if submit && !accepted {
            std::process::exit(1);
        }
        return Ok(());
    }

    let headline = if solved {
        "solved"
    } else if accepted {
        "accepted, star unconfirmed"
    } else {
        "recorded"
    };
    println!("{headline} {id} in {} turn(s)", report.turns);

    if !report.history.is_empty() {
        println!("turn timings:");
        for turn in &report.history {
            println!(
                "  turn {}: ttft {}, total {}, {:.1} KB, tools {} ({} err), tests {}/{}, {} tok",
                turn.turn,
                turn.ttft_ms
                    .map(format_seconds)
                    .unwrap_or_else(|| "-".to_string()),
                format_seconds(turn.duration_ms),
                turn.reply_bytes as f64 / 1024.0,
                turn.tool_calls,
                turn.tool_errors,
                turn.tests_passed,
                turn.tests_total,
                turn.total_tokens
            );
        }
    }
    if report.turns > 0 {
        if report.verified {
            println!(
                "example tests: {}/{} passed",
                report.tests.passed(),
                report.tests.cases.len()
            );
        } else {
            warning("no example tests were available; the answer is unverified");
        }
    }
    for run in &report.runs {
        println!(
            "part {}: {} ({} ms, {})",
            run.part,
            run.answer,
            run.millis,
            match run.verdict {
                Verdict::Correct => "matches the recorded answer",
                Verdict::Incorrect => "does NOT match the recorded answer",
                Verdict::Unverified => "recorded, no known answer",
            }
        );
    }
    for submission in &report.submissions {
        println!(
            "submit part {}: {} — {}",
            submission.part, submission.status, submission.message
        );
    }
    if !report.recorded.is_empty() {
        println!(
            "recorded answer(s): {}",
            report
                .recorded
                .iter()
                .map(|part| format!("part {part}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(note) = &report.note {
        println!("note: {note}");
    }
    if let Some((state, _)) = &star {
        println!(
            "remote: {} star(s); part 1 {}; part 2 {}",
            state.stars,
            if state.part1.solved {
                "solved"
            } else {
                "unsolved"
            },
            if state.part2.solved {
                "solved"
            } else if state.part2_unlocked {
                "unlocked, unsolved"
            } else {
                "locked"
            },
        );
    }
    println!(
        "code: {}",
        solutions
            .join(id.year.to_string())
            .join(format!("{:02}", id.day))
            .join("solve.rs")
            .display()
    );
    if submit && !accepted {
        std::process::exit(1);
    }
    Ok(())
}

fn format_seconds(millis: u64) -> String {
    format!("{:.1}s", millis as f64 / 1000.0)
}

fn has_input(archive: &Path, year: u32, day: u32) -> bool {
    aoc::input_path(archive, TaskId::new(year, day)).is_file()
}

fn find_input(archive: &Path, id: TaskId) -> Option<PathBuf> {
    let path = aoc::input_path(archive, id);
    path.is_file().then_some(path)
}

fn solution_exists(solutions: &Path, year: u32, day: u32) -> bool {
    let dir = solutions.join(year.to_string()).join(format!("{day:02}"));
    ["solve", "solve.py", "solve.rs"]
        .iter()
        .any(|name| dir.join(name).is_file())
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn default_years() -> Vec<u32> {
    let (year, month) = aoc::current_year_month();
    let last = if month >= 12 {
        year
    } else {
        year.saturating_sub(1)
    };
    (2015..=last.max(2015)).collect()
}

const RUST_TEMPLATE: &str = r#"use std::io::{self, Read};

fn part1(input: &str) -> String {
    let _ = input;
    todo!()
}

fn part2(input: &str) -> String {
    let _ = input;
    todo!()
}

fn main() {
    let part: u8 = std::env::var("AOC_PART")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);

    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("reading stdin");

    let answer = match part {
        1 => part1(&input),
        _ => part2(&input),
    };
    println!("{answer}");
}
"#;
