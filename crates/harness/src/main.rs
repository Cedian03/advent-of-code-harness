use harness::agent::{Agent, AgentEvent};
use harness::aoc::{self, Aoc, ArchiveAoc, TaskId, Verdict, tests_from_puzzle};
use harness::model::{Model, OpenAiModel, ReplayModel};
use harness::runner;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
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
    /// Report archive coverage
    Status,
    /// Print a captured statement as Markdown
    Show { year: u32, day: u32 },
    /// List days for a year with solution/input status
    List(ListArgs),
    /// Scaffold a solution for a day
    Init(InitArgs),
    /// Import your personal puzzle input into the archive
    Input(InputArgs),
    /// Emit a machine-readable task bundle (statement, tests, paths)
    Task { year: u32, day: u32 },
    /// Run the solution against the detected example tests
    Test(TestArgs),
    /// Run the solution against your real puzzle input
    Run(RunArgs),
    /// Drive an LLM loop that writes, tests, and runs a solution
    Solve(SolveArgs),
    /// Pick the next unsolved day
    Next(NextArgs),
}

#[derive(Args)]
struct ScrapeArgs {
    /// Years to scrape: "2015-2025", "2024,2019", or a single year.
    /// Defaults to every past event (and the current one in December).
    #[arg(long)]
    years: Option<String>,

    /// Days to scrape: "1-25" or a single day.
    #[arg(long, default_value = "1-25")]
    days: String,

    /// Minimum seconds between HTTP requests. AoC asks for at most one per second.
    #[arg(long, default_value_t = 1.1)]
    delay: f64,

    /// Refetch pages even when they are already cached.
    #[arg(long)]
    refresh: bool,

    /// Re-parse cached pages without refetching (after parser changes).
    #[arg(long)]
    reparse: bool,

    /// Environment variable holding your adventofcode.com session cookie.
    #[arg(long, default_value = "AOC_SESSION")]
    cookie_env: String,
}

#[derive(Args)]
struct ListArgs {
    #[arg(long)]
    year: Option<u32>,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct InitArgs {
    year: u32,
    day: u32,

    /// Copy a puzzle input into the archive (never committed).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Overwrite an existing solve.rs.
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct InputArgs {
    year: u32,
    day: u32,

    /// Read the input from this file instead of stdin.
    #[arg(long)]
    file: Option<PathBuf>,
}

#[derive(Args)]
struct TestArgs {
    year: u32,
    day: u32,

    /// Override the solution command (sh -c; input on stdin, AOC_PART set).
    #[arg(long)]
    cmd: Option<String>,

    /// Only test one part.
    #[arg(long)]
    part: Option<u8>,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct NextArgs {
    #[arg(long)]
    year: Option<u32>,

    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct SolveArgs {
    year: u32,
    day: u32,

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
    year: u32,
    day: u32,

    /// Override the solution command (sh -c; input on stdin, AOC_PART set).
    #[arg(long)]
    cmd: Option<String>,

    /// Which part to run (default: 1).
    #[arg(long, default_value_t = 1)]
    part: u8,

    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
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
    match cli.command {
        Command::Scrape(args) => scrape_command(args, &cli.archive),
        Command::Status => status_command(&cli.archive),
        Command::Show { year, day } => show_command(&cli.archive, year, day),
        Command::List(args) => list_command(args, &cli.archive, &cli.solutions),
        Command::Init(args) => init_command(args, &cli.archive, &cli.solutions),
        Command::Input(args) => input_command(args, &cli.archive),
        Command::Task { year, day } => task_command(year, day, &cli.archive, &cli.solutions),
        Command::Test(args) => test_command(args, &cli.archive, &cli.solutions, &cli.build_dir),
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
    let years = match &args.years {
        Some(spec) => parse_years(spec)?,
        None => default_years(),
    };
    let days = parse_days(&args.days)?;

    if args.delay < 1.0 {
        bail!(
            "--delay must be at least 1.0 seconds; AoC asks clients to make at most one request per second"
        );
    }

    let cookie = std::env::var(&args.cookie_env).ok();
    let root = scrape::run(&ScrapeOptions {
        years,
        days,
        out_dir: archive.to_path_buf(),
        delay: Duration::from_secs_f64(args.delay),
        refresh: args.refresh,
        reparse: args.reparse,
        cookie,
    })?;

    let total: usize = root.years.iter().map(|y| y.scraped_days()).sum();
    let part2: usize = root
        .years
        .iter()
        .flat_map(|y| &y.days)
        .filter(|d| d.parts_captured >= 2)
        .count();
    let inputs: usize = root
        .years
        .iter()
        .flat_map(|year| &year.days)
        .filter(|entry| entry.input.is_some())
        .count();
    eprintln!(
        "\nArchive: {total} day(s) across {} year(s); {part2} with Part Two; {inputs} with inputs.",
        root.years.len()
    );
    Ok(())
}

fn status_command(archive: &Path) -> Result<()> {
    let years = scrape::available_years(archive);
    if years.is_empty() {
        bail!(
            "no archive in {} yet — run `harness scrape` first",
            archive.display()
        );
    }

    println!(
        "{:<6} {:>5} {:>9} {:>8} {:>7} {:>6}",
        "Year", "Days", "Captured", "Part 2", "Inputs", "Tests"
    );
    let mut days = 0;
    let mut captured = 0;
    let mut part2 = 0;
    let mut inputs = 0;
    let mut tests = 0;
    for year in &years {
        let year_days = year.days.len();
        let year_captured = year.scraped_days();
        let year_part2 = year.days.iter().filter(|d| d.parts_captured >= 2).count();
        let year_inputs = year
            .days
            .iter()
            .filter(|day| has_input(archive, year.year, day.day))
            .count();
        let year_tests: u32 = year.days.iter().map(|d| d.tests).sum();
        println!(
            "{:<6} {:>5} {:>9} {:>8} {:>7} {:>6}",
            year.year, year_days, year_captured, year_part2, year_inputs, year_tests
        );
        days += year_days;
        captured += year_captured;
        part2 += year_part2;
        inputs += year_inputs;
        tests += year_tests;
    }
    println!(
        "\n{} day(s) on the calendar, {} captured, {} with Part Two, {} with inputs, {} detected example tests.",
        days, captured, part2, inputs, tests
    );
    if part2 == 0 && inputs == 0 {
        println!(
            "Set AOC_SESSION and re-run `harness scrape` to add Part Two statements, inputs, and your past answers."
        );
    }
    Ok(())
}

fn show_command(archive: &Path, year: u32, day: u32) -> Result<()> {
    let path = scrape::puzzle_dir(archive, year, day).join("puzzle.md");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("no captured puzzle at {}", path.display()))?;
    print!("{text}");
    Ok(())
}

fn list_command(args: ListArgs, archive: &Path, solutions: &Path) -> Result<()> {
    let years = scrape::available_years(archive);
    let year = match args.year {
        Some(year) => year,
        None => years
            .last()
            .map(|index| index.year)
            .context("no archive yet — run `harness scrape` first")?,
    };
    let index = years
        .iter()
        .find(|index| index.year == year)
        .with_context(|| format!("no archived data for {year}"))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(index)?);
        return Ok(());
    }

    println!(
        "{:<3} {:<44} {:>5} {:>3} {:>6} {:>9}",
        "Day", "Title", "Tests", "P2", "Input", "Solution"
    );
    for entry in &index.days {
        let title = entry.title.as_deref().unwrap_or("-");
        println!(
            "{:<3} {:<44} {:>5} {:>3} {:>6} {:>9}",
            entry.day,
            truncate(title, 44),
            entry.tests,
            if entry.parts_captured >= 2 {
                "yes"
            } else {
                "no"
            },
            if has_input(archive, year, entry.day) {
                "yes"
            } else {
                "no"
            },
            if solution_exists(solutions, year, entry.day) {
                "yes"
            } else {
                "no"
            }
        );
    }
    Ok(())
}

fn init_command(args: InitArgs, archive: &Path, solutions: &Path) -> Result<()> {
    let dir = solutions
        .join(args.year.to_string())
        .join(format!("{:02}", args.day));
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let solved = ["solve", "solve.py", "solve.rs"]
        .iter()
        .any(|name| dir.join(name).is_file());
    let source = dir.join("solve.rs");
    if args.force || !solved {
        fs::write(&source, RUST_TEMPLATE)
            .with_context(|| format!("writing {}", source.display()))?;
        println!("created {}", source.display());
    } else {
        println!("solution already exists in {}", dir.display());
    }

    if let Some(input) = args.input {
        let target = scrape::puzzle_dir(archive, args.year, args.day).join("input.txt");
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&input, &target)
            .with_context(|| format!("copying {} to {}", input.display(), target.display()))?;
        println!("imported input -> {}", target.display());
    }

    if !scrape::puzzle_dir(archive, args.year, args.day)
        .join("puzzle.json")
        .is_file()
    {
        println!(
            "note: no archived statement for {}/{} — run `harness scrape --years {} --days {}`",
            args.year, args.day, args.year, args.day
        );
    }

    println!("next: harness test {} {}", args.year, args.day);
    Ok(())
}

fn input_command(args: InputArgs, archive: &Path) -> Result<()> {
    if !(1..=25).contains(&args.day) {
        bail!("day must be within 1-25");
    }

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
        bail!("input is empty; nothing imported");
    }

    let id = TaskId::new(args.year, args.day);
    let path = write_input(archive, id, &bytes)?;
    println!("imported {} bytes -> {}", bytes.len(), path.display());
    if !scrape::puzzle_dir(archive, args.year, args.day)
        .join("puzzle.json")
        .is_file()
    {
        println!(
            "note: no archived puzzle for {id} yet — run `harness scrape --years {} --days {}`",
            args.year, args.day
        );
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

fn task_command(year: u32, day: u32, archive: &Path, solutions: &Path) -> Result<()> {
    let puzzle = load_puzzle(archive, year, day)?;
    let dir = scrape::puzzle_dir(archive, year, day);
    let statement = fs::read_to_string(dir.join("puzzle.md")).unwrap_or_default();
    let id = TaskId::new(year, day);
    let input = find_input(archive, id);
    let tests = tests_from_puzzle(&puzzle);

    let bundle = serde_json::json!({
        "year": year,
        "day": day,
        "title": puzzle.title,
        "url": puzzle.url,
        "part_two_captured": puzzle.is_complete(),
        "statement": statement,
        "input_path": input.as_ref().map(|path| path.to_string_lossy()),
        "solution_dir": solutions.join(year.to_string()).join(format!("{day:02}")),
        "tests": tests,
    });
    println!("{}", serde_json::to_string_pretty(&bundle)?);
    Ok(())
}

fn test_command(args: TestArgs, archive: &Path, solutions: &Path, build_dir: &Path) -> Result<()> {
    let puzzle = load_puzzle(archive, args.year, args.day)?;
    let mut tests = tests_from_puzzle(&puzzle);
    if let Some(part) = args.part {
        tests.retain(|test| test.part == part);
    }
    if tests.is_empty() {
        println!(
            "{}/{:02} has no detected example tests{}.",
            args.year,
            args.day,
            if puzzle.is_complete() {
                ""
            } else {
                " (Part Two is not in the archive)"
            }
        );
        return Ok(());
    }

    let solution = runner::discover(
        solutions,
        build_dir,
        args.year,
        args.day,
        args.cmd.as_deref(),
    )?;

    let mut results = Vec::new();
    let mut failures = 0usize;
    for test in &tests {
        let outcome = runner::execute(&solution, args.year, args.day, test.part, &test.input)?;
        let actual = outcome.stdout.trim().to_string();
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
        println!(
            "{}/{:02} {} — {} test(s)",
            args.year,
            args.day,
            puzzle.title,
            tests.len()
        );
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
    if !(1..=2).contains(&args.part) {
        bail!("--part must be 1 or 2");
    }

    let puzzle = load_puzzle(archive, args.year, args.day)?;
    let input_path = find_input(archive, TaskId::new(args.year, args.day)).with_context(|| {
        format!(
            "no puzzle input for {}/{:02}\n\
             Put it at {} (e.g. `harness init {} {} --input path/to/input.txt`),\n\
             or set AOC_SESSION and run `harness scrape --years {} --refresh`.",
            args.year,
            args.day,
            scrape::puzzle_dir(archive, args.year, args.day)
                .join("input.txt")
                .display(),
            args.year,
            args.day,
            args.year
        )
    })?;
    let input = fs::read_to_string(&input_path)
        .with_context(|| format!("reading {}", input_path.display()))?;

    let solution = runner::discover(
        solutions,
        build_dir,
        args.year,
        args.day,
        args.cmd.as_deref(),
    )?;
    let outcome = runner::execute(&solution, args.year, args.day, args.part, &input)?;

    let answer = outcome.stdout.trim().to_string();
    let archived = puzzle
        .answers
        .get((args.part - 1) as usize)
        .and_then(|answer| answer.as_deref());
    let matches = archived.map(|expected| expected.trim() == answer);

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "part": args.part,
                "answer": answer,
                "archived_answer": archived,
                "matches_archive": matches,
                "success": outcome.success,
                "stderr": outcome.stderr.trim(),
                "millis": outcome.elapsed.as_millis() as u64,
            }))?
        );
    } else {
        if outcome.success {
            println!(
                "{}/{:02} part {}: {}  ({} ms)",
                args.year,
                args.day,
                args.part,
                if answer.is_empty() {
                    "<no output>"
                } else {
                    &answer
                },
                outcome.elapsed.as_millis()
            );
        } else {
            println!(
                "{}/{:02} part {} failed after {} ms",
                args.year,
                args.day,
                args.part,
                outcome.elapsed.as_millis()
            );
        }
        if !outcome.stderr.trim().is_empty() {
            for line in outcome.stderr.trim().lines().take(10) {
                eprintln!("  {line}");
            }
        }
        match matches {
            Some(true) => println!("  matches the archived answer"),
            Some(false) => println!(
                "  does not match the archived answer ({})",
                archived.unwrap_or("")
            ),
            None => {}
        }
    }

    if !outcome.success {
        std::process::exit(1);
    }
    Ok(())
}

fn next_command(args: NextArgs, archive: &Path, solutions: &Path, build_dir: &Path) -> Result<()> {
    let local = ArchiveAoc::new(
        archive.to_path_buf(),
        solutions.to_path_buf(),
        build_dir.to_path_buf(),
    );
    let tasks = local.tasks()?;
    let next = tasks
        .iter()
        .filter(|task| args.year.is_none_or(|year| year == task.id.year))
        .filter(|task| task.solved.len() < task.parts as usize)
        .min_by_key(|task| (!task.has_input, task.id.year, task.id.day));

    let Some(task) = next else {
        bail!(
            "every captured part is solved{}",
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
        println!("run: harness solve {} {}", task.id.year, task.id.day);
    }
    Ok(())
}

fn solve_command(
    args: SolveArgs,
    archive: &Path,
    solutions: &Path,
    build_dir: &Path,
) -> Result<()> {
    if !(1..=25).contains(&args.day) {
        bail!("day must be within 1-25");
    }
    if args.turns == 0 {
        bail!("--turns must be at least 1");
    }
    if args.timeout == 0 {
        bail!("--timeout must be at least 1 second");
    }

    let local = ArchiveAoc::new(
        archive.to_path_buf(),
        solutions.to_path_buf(),
        build_dir.to_path_buf(),
    );
    let id = TaskId::new(args.year, args.day);

    let mut emit = |event: &AgentEvent| {
        if args.events {
            match serde_json::to_string(event) {
                Ok(line) => eprintln!("{line}"),
                Err(err) => tracing::warn!(%err, "could not serialize agent event"),
            }
        }
    };

    let report = if let Some(replay) = &args.replay {
        let model = ReplayModel::from_file(replay)?;
        if !args.quiet {
            eprintln!("replay: {}", replay.display());
        }
        let agent = Agent::new(&local, &model, args.turns);
        (if args.quiet { agent.quiet() } else { agent }).solve_with(id, &mut emit)?
    } else {
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
        let agent = Agent::new(&local, &model, args.turns);
        (if args.quiet { agent.quiet() } else { agent }).solve_with(id, &mut emit)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let verified = report.verified
        || report
            .runs
            .iter()
            .any(|run| run.verdict == Verdict::Correct);
    println!(
        "{} {} in {} turn(s)",
        if verified { "solved" } else { "finished" },
        id,
        report.turns
    );
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
    if report.verified {
        println!(
            "example tests: {}/{} passed",
            report.tests.passed(),
            report.tests.cases.len()
        );
    } else {
        println!("warning: no example tests were available; the answer is unverified");
    }
    for run in &report.runs {
        println!(
            "part {}: {} ({} ms, {})",
            run.part,
            run.answer,
            run.millis,
            match run.verdict {
                Verdict::Correct => "matches archived answer",
                Verdict::Incorrect => "does NOT match archived answer",
                Verdict::Unverified => "recorded, no archived answer",
            }
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
    println!(
        "code: {}",
        solutions
            .join(args.year.to_string())
            .join(format!("{:02}", args.day))
            .join("solve.rs")
            .display()
    );
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

fn parse_years(spec: &str) -> Result<Vec<u32>> {
    let mut years = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((from, to)) = part.split_once('-') {
            let from: u32 = from.trim().parse().context("invalid year range start")?;
            let to: u32 = to.trim().parse().context("invalid year range end")?;
            if from > to {
                bail!("invalid year range {from}-{to}");
            }
            years.extend(from..=to);
        } else {
            years.push(part.parse().context("invalid year")?);
        }
    }
    years.sort_unstable();
    years.dedup();
    if years.is_empty() {
        bail!("no years given");
    }
    Ok(years)
}

fn parse_days(spec: &str) -> Result<Option<(u32, u32)>> {
    let spec = spec.trim();
    if spec == "all" || spec == "1-25" {
        return Ok(None);
    }
    let (from, to) = match spec.split_once('-') {
        Some((from, to)) => (from.trim(), to.trim()),
        None => (spec, spec),
    };
    let from: u32 = from.parse().context("invalid day")?;
    let to: u32 = to.parse().context("invalid day")?;
    if from == 0 || to > 25 || from > to {
        bail!("day range must be within 1-25, got {spec}");
    }
    Ok(Some((from, to)))
}

fn default_years() -> Vec<u32> {
    let (year, month) = current_year_month();
    let last = if month >= 12 {
        year
    } else {
        year.saturating_sub(1)
    };
    (2015..=last.max(2015)).collect()
}

fn current_year_month() -> (u32, u32) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let (year, month, _) = civil_from_days(days);
    (year as u32, month)
}

/// Howard Hinnant's civil_from_days, without pulling in a date crate.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
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
