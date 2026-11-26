use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::debug;

use crate::client::Fetcher;
use crate::html::{self, now_unix};
use crate::model::{Puzzle, RootIndex, YearIndex};

pub struct ScrapeOptions {
    pub years: Vec<u32>,
    /// Exact `(year, day)` pairs to visit, or `None` for every Day of `years`.
    /// A Selection arrives here already deduplicated and sorted.
    pub days: Option<BTreeSet<(u32, u32)>>,
    pub out_dir: PathBuf,
    pub delay: Duration,
    pub refresh: bool,
    /// Re-parse cached raw pages even if a puzzle.json already exists.
    pub reparse: bool,
    pub cookie: Option<String>,
}

pub fn run(opts: &ScrapeOptions) -> Result<RootIndex> {
    fs::create_dir_all(&opts.out_dir)
        .with_context(|| format!("creating {}", opts.out_dir.display()))?;

    let mut fetcher = Fetcher::new(
        opts.out_dir.clone(),
        opts.cookie.clone(),
        opts.delay,
        opts.refresh,
    );

    for &year in &opts.years {
        if let Err(err) = scrape_year(&mut fetcher, year, opts) {
            eprintln!("[{year}] skipped: {err:#}");
        }
    }

    write_root_index(&opts.out_dir)
}

fn scrape_year(fetcher: &mut Fetcher, year: u32, opts: &ScrapeOptions) -> Result<()> {
    let calendar_html = fetcher.calendar(year)?;
    let mut index = html::parse_calendar(year, &calendar_html);

    if index.days.is_empty() {
        anyhow::bail!("no days found on the {year} calendar");
    }
    eprintln!("[{year}] {} days on the calendar", index.days.len());

    let mut captured = 0usize;
    for day in index
        .days
        .iter_mut()
        .map(|entry| entry.day)
        .collect::<Vec<_>>()
    {
        if let Some(days) = &opts.days
            && !days.contains(&(year, day))
        {
            continue;
        }

        let day_dir = opts
            .out_dir
            .join(year.to_string())
            .join(format!("{day:02}"));
        let puzzle_path = day_dir.join("puzzle.json");

        let cached = (!opts.refresh && !opts.reparse)
            .then(|| load_puzzle_if_exists(&puzzle_path))
            .flatten();

        let refetch = cached
            .as_ref()
            .is_none_or(|puzzle| needs_refetch(fetcher, puzzle));

        let puzzle = if refetch {
            // A session added after the public scrape can reveal Part Two,
            // but the cached page is the public one. Refetch it in that case.
            let force_page = fetcher.authenticated()
                && cached.as_ref().is_some_and(|puzzle| !puzzle.authenticated);
            let page = match if force_page {
                fetcher.day_fresh(year, day)
            } else {
                fetcher.day(year, day)
            } {
                Ok(page) => page,
                Err(err) => {
                    eprintln!("[{year}] day {day:02} failed: {err:#}");
                    continue;
                }
            };
            let input = fetcher.input(year, day).unwrap_or_else(|err| {
                eprintln!("[{year}] day {day:02} input failed: {err:#}");
                None
            });
            let relative_input = input.as_ref().map(|_| format!("{year}/{day:02}/input.txt"));

            let puzzle = html::parse_day(year, day, &page, fetcher.authenticated(), relative_input);

            fs::create_dir_all(&day_dir)
                .with_context(|| format!("creating {}", day_dir.display()))?;
            fs::write(&puzzle_path, serde_json::to_string_pretty(&puzzle)?)
                .with_context(|| format!("writing {}", puzzle_path.display()))?;
            fs::write(day_dir.join("puzzle.md"), render_puzzle_md(&puzzle))
                .with_context(|| format!("writing {}/puzzle.md", day_dir.display()))?;
            if let Some(input) = input {
                fs::write(day_dir.join("input.txt"), input)
                    .with_context(|| format!("writing {}/input.txt", day_dir.display()))?;
            }
            puzzle
        } else {
            cached.expect("refetch is false only when a cached puzzle exists")
        };

        if let Some(entry) = index.days.iter_mut().find(|entry| entry.day == day) {
            entry.title = Some(puzzle.title.clone());
            entry.scraped = true;
            entry.parts_captured = puzzle.parts.len() as u8;
            entry.tests = puzzle
                .parts
                .iter()
                .flat_map(|part| &part.examples)
                .filter(|example| example.expected.is_some())
                .count() as u32;
            entry.input = puzzle.input.clone();
        }

        captured += 1;
        debug!(
            year,
            day,
            parts = puzzle.parts.len(),
            examples = puzzle
                .parts
                .iter()
                .map(|part| part.examples.len())
                .sum::<usize>(),
            "parsed day"
        );
        eprintln!(
            "[{year}] day {day:02} {:<40} {} part(s), {} example(s), {}{}",
            truncate(&puzzle.title, 40),
            puzzle.parts.len(),
            puzzle.parts.iter().map(|p| p.examples.len()).sum::<usize>(),
            if puzzle.input.is_some() {
                "input"
            } else {
                "no input"
            },
            if puzzle.answers.iter().any(Option::is_some) {
                ", answers recorded"
            } else {
                ""
            }
        );
    }

    // Days skipped by a --days filter still live on disk. Rebuild the index
    // from the archive so a partial run cannot mark existing days unscraped.
    refresh_index_from_disk(&opts.out_dir, year, &mut index);

    index.days.sort_by_key(|entry| entry.day);
    write_year_index(&opts.out_dir, &index)?;
    let inputs = index
        .days
        .iter()
        .filter(|entry| entry.input.is_some())
        .count();
    eprintln!("[{year}] captured {captured} day(s), {inputs} with input");
    Ok(())
}

fn needs_refetch(fetcher: &Fetcher, puzzle: &Puzzle) -> bool {
    fetcher.authenticated()
        && (!puzzle.authenticated || !puzzle.is_complete() || puzzle.input.is_none())
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn render_puzzle_md(puzzle: &Puzzle) -> String {
    let mut md = String::new();
    md.push_str(&format!(
        "# {} Day {}: {}\n\nSource: <{}>\n\n",
        puzzle.year, puzzle.day, puzzle.title, puzzle.url
    ));
    md.push_str(&format!("Scraped: {} (unix)\n\n", puzzle.fetched_at));

    if puzzle.authenticated {
        md.push_str("Captured with a logged-in session.\n\n");
    } else {
        md.push_str(
            "> Captured from the public page. Part Two is only revealed by AoC after Part One \
             is solved, so it is absent here.\n\n",
        );
    }

    for part in &puzzle.parts {
        let name = match part.number {
            1 => "One",
            _ => "Two",
        };
        md.push_str(&format!("## Part {name}\n\n"));
        md.push_str(&strip_leading_heading(&part.markdown));
        md.push_str("\n\n");

        if part.examples.iter().any(|e| e.expected.is_some()) {
            md.push_str("### Example tests (auto-detected)\n\n");
            let mut shown = 0;
            for example in &part.examples {
                let Some(expected) = &example.expected else {
                    continue;
                };
                shown += 1;
                md.push_str(&format!(
                    "**Example {shown}**\n\nInput:\n\n```\n{}\n```\n\nExpected output: `{expected}`\n\n",
                    example.input
                ));
            }
        }
    }

    if !puzzle.is_complete() {
        md.push_str(
            "## Part Two\n\n_Not captured: AoC reveals Part Two after Part One is solved with a \
             logged-in session._\n",
        );
    }

    md
}

/// Drop the article's own "## Day N: Title" heading, which now duplicates the
/// document title.
fn strip_leading_heading(markdown: &str) -> String {
    let markdown = markdown.trim_start();
    if let Some(rest) = markdown.strip_prefix("## ") {
        if let Some((_, body)) = rest.split_once('\n') {
            return body.trim_start().to_string();
        }
        return String::new();
    }
    markdown.to_string()
}

fn refresh_index_from_disk(out_dir: &Path, year: u32, index: &mut YearIndex) {
    for entry in &mut index.days {
        let day_dir = out_dir
            .join(year.to_string())
            .join(format!("{:02}", entry.day));
        let Some(puzzle) = load_puzzle_if_exists(&day_dir.join("puzzle.json")) else {
            continue;
        };
        entry.title = Some(puzzle.title.clone());
        entry.scraped = true;
        entry.parts_captured = puzzle.parts.len() as u8;
        entry.tests = puzzle
            .parts
            .iter()
            .flat_map(|part| &part.examples)
            .filter(|example| example.expected.is_some())
            .count() as u32;
        entry.input = day_dir
            .join("input.txt")
            .is_file()
            .then(|| format!("{year}/{:02}/input.txt", entry.day));
    }
}

fn load_puzzle_if_exists(path: &Path) -> Option<Puzzle> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn puzzle_dir(out_dir: &Path, year: u32, day: u32) -> PathBuf {
    out_dir.join(year.to_string()).join(format!("{day:02}"))
}

pub fn load_puzzle(out_dir: &Path, year: u32, day: u32) -> Result<Puzzle> {
    let path = puzzle_dir(out_dir, year, day).join("puzzle.json");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("no archived puzzle at {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn load_year_index(out_dir: &Path, year: u32) -> Result<YearIndex> {
    let path = out_dir.join(year.to_string()).join("index.json");
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn write_year_index(out_dir: &Path, index: &YearIndex) -> Result<()> {
    let dir = out_dir.join(index.year.to_string());
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("index.json"), serde_json::to_string_pretty(index)?)
        .with_context(|| format!("writing {}/index.json", dir.display()))
}

pub fn write_root_index(out_dir: &Path) -> Result<RootIndex> {
    let mut years = Vec::new();
    if let Ok(entries) = fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let Ok(year) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if let Ok(index) = load_year_index(out_dir, year) {
                years.push(index);
            }
        }
    }
    years.sort_by_key(|index| index.year);

    let root = RootIndex {
        generated_at: now_unix(),
        years,
    };
    fs::write(
        out_dir.join("index.json"),
        serde_json::to_string_pretty(&root)?,
    )
    .with_context(|| format!("writing {}/index.json", out_dir.display()))?;
    Ok(root)
}

pub fn available_years(out_dir: &Path) -> Vec<YearIndex> {
    write_root_index(out_dir)
        .map(|root| root.years)
        .unwrap_or_default()
}
