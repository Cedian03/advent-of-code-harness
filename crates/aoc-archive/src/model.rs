use serde::{Deserialize, Serialize};

/// A single captured puzzle example: the sample input shown in a code block
/// plus the answer the prose says it should produce, when one can be found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
}

/// One half ("part") of a day's puzzle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    /// 1 or 2.
    pub number: u8,
    /// Raw heading text, e.g. "--- Day 1: Historian Hysteria ---".
    pub heading: String,
    /// Rendered statement as Markdown (the form the harness will feed to solvers).
    pub markdown: String,
    /// Raw HTML of the article, kept so parsing can be improved without refetching.
    pub html: String,
    pub examples: Vec<Example>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Puzzle {
    pub year: u32,
    pub day: u32,
    pub title: String,
    pub url: String,
    /// Unix seconds when this record was scraped.
    pub fetched_at: u64,
    /// Whether the request that produced this record carried a session cookie.
    pub authenticated: bool,
    pub parts: Vec<Part>,
    /// Answers AoC reported as already submitted, per part (session-only).
    pub answers: Vec<Option<String>>,
    /// Relative path to the puzzle input, when a session cookie was available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl Puzzle {
    /// Part Two is only rendered by AoC once Part One has been solved, so a
    /// public scrape can legitimately contain just one part.
    pub fn is_complete(&self) -> bool {
        self.parts.iter().any(|p| p.number == 2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayEntry {
    pub day: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// False when the day exists on the calendar but has not been captured yet.
    pub scraped: bool,
    pub parts_captured: u8,
    /// Number of examples with a detected expected output (usable as tests).
    #[serde(default)]
    pub tests: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearIndex {
    pub year: u32,
    pub url: String,
    pub days: Vec<DayEntry>,
}

impl YearIndex {
    pub fn scraped_days(&self) -> usize {
        self.days.iter().filter(|d| d.scraped).count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootIndex {
    pub generated_at: u64,
    pub years: Vec<YearIndex>,
}
