use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tracing::debug;

/// How the harness will invoke a solution.
#[derive(Debug)]
pub enum Runner {
    /// `sh -c <script>` — used for `--cmd` and scripted languages.
    Shell(String),
    /// A program invoked with the given arguments.
    Program { program: PathBuf, args: Vec<String> },
}

#[derive(Debug)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub elapsed: Duration,
}

/// Locate a solution for `year/day`.
///
/// Contract: the program reads the puzzle input on stdin, writes the answer on
/// stdout, and reads the part number (1 or 2) from `AOC_PART`.
///
/// Priority: `--cmd`, then `solve` (executable), `solve.py`, `solve.rs`.
pub fn discover(
    solutions: &Path,
    build_dir: &Path,
    year: u32,
    day: u32,
    command: Option<&str>,
) -> Result<Runner> {
    if let Some(script) = command {
        return Ok(Runner::Shell(script.to_string()));
    }

    let dir = solutions.join(year.to_string()).join(format!("{day:02}"));
    if !dir.is_dir() {
        bail!(
            "no solution directory at {} — run `harness init {year} {day}` first",
            dir.display()
        );
    }

    let executable = dir.join("solve");
    if executable.is_file() {
        return Ok(Runner::Program {
            program: executable,
            args: Vec::new(),
        });
    }

    let python = dir.join("solve.py");
    if python.is_file() {
        return Ok(Runner::Program {
            program: PathBuf::from("python3"),
            args: vec![python.to_string_lossy().into_owned()],
        });
    }

    let rust = dir.join("solve.rs");
    if rust.is_file() {
        let binary = compile_rust(&rust, build_dir, year, day)?;
        return Ok(Runner::Program {
            program: binary,
            args: Vec::new(),
        });
    }

    bail!(
        "no solution in {} — expected `solve`, `solve.py`, or `solve.rs` (or pass --cmd)",
        dir.display()
    )
}

pub fn execute(runner: &Runner, year: u32, day: u32, part: u8, input: &str) -> Result<Outcome> {
    let mut command = match runner {
        Runner::Shell(script) => {
            let mut command = Command::new("sh");
            command.arg("-c").arg(script);
            command
        }
        Runner::Program { program, args } => {
            let mut command = Command::new(program);
            command.args(args);
            command
        }
    };

    command
        .env("AOC_PART", part.to_string())
        .env("AOC_YEAR", year.to_string())
        .env("AOC_DAY", day.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    debug!(?runner, part, input_bytes = input.len(), "running solution");
    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("spawning {runner:?}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        // A solver is allowed to exit without draining stdin; that is not an
        // error we care about, the answer on stdout is what matters.
        let _ = stdin.write_all(input.as_bytes());
    }

    let output = child
        .wait_with_output()
        .with_context(|| format!("running {runner:?}"))?;
    let elapsed = started.elapsed();
    debug!(
        success = output.status.success(),
        millis = elapsed.as_millis() as u64,
        stdout_bytes = output.stdout.len(),
        stderr_bytes = output.stderr.len(),
        "solution finished"
    );

    Ok(Outcome {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        success: output.status.success(),
        elapsed,
    })
}

/// Build a Rust source file and return a runner for the resulting binary.
pub fn rust(source: &Path, build_dir: &Path, year: u32, day: u32) -> Result<Runner> {
    let binary = compile_rust(source, build_dir, year, day)?;
    Ok(Runner::Program {
        program: binary,
        args: Vec::new(),
    })
}

fn compile_rust(source: &Path, build_dir: &Path, year: u32, day: u32) -> Result<PathBuf> {
    let dir = build_dir.join(year.to_string()).join(format!("{day:02}"));
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let binary = dir.join("solve");

    if !is_stale(source, &binary) {
        return Ok(binary);
    }

    eprintln!("compiling {}", source.display());
    let output = Command::new("rustc")
        .arg("-O")
        .arg("--edition")
        .arg("2024")
        .arg("-o")
        .arg(&binary)
        .arg(source)
        .output()
        .context("running rustc (is it installed?)")?;

    if !output.status.success() {
        bail!(
            "rustc failed:\n{}",
            String::from_utf8_lossy(&output.stderr).trim_end()
        );
    }
    Ok(binary)
}

fn is_stale(source: &Path, binary: &Path) -> bool {
    let Ok(binary_time) = fs::metadata(binary).and_then(|meta| meta.modified()) else {
        return true;
    };
    let Ok(source_time) = fs::metadata(source).and_then(|meta| meta.modified()) else {
        return true;
    };
    source_time > binary_time
}
