use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

use crate::model::Puzzle;
use crate::scrape;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemotePartState {
    pub solved: bool,
    pub answer: Option<String>,
}

/// What adventofcode.com currently knows about one day for this account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDayState {
    pub year: u32,
    pub day: u32,
    /// 0, 1, or 2 stars.
    pub stars: u8,
    pub part1: RemotePartState,
    pub part2: RemotePartState,
    pub part2_unlocked: bool,
    /// Unix seconds when this snapshot was taken.
    pub fetched_at: u64,
}

impl RemoteDayState {
    pub fn new(year: u32, day: u32) -> Self {
        Self {
            year,
            day,
            stars: 0,
            part1: RemotePartState::default(),
            part2: RemotePartState::default(),
            part2_unlocked: false,
            fetched_at: now(),
        }
    }

    pub fn from_puzzle(year: u32, day: u32, puzzle: &Puzzle, stars: u8) -> Self {
        let part2_present = puzzle.parts.iter().any(|part| part.number == 2);
        let answer1 = puzzle.answers.first().cloned().flatten();
        let answer2 = puzzle.answers.get(1).cloned().flatten();

        let part1_solved = answer1.is_some() || part2_present || stars >= 1;
        let part2_solved = answer2.is_some() || stars >= 2;
        let stars = stars.max(u8::from(part1_solved) + u8::from(part2_solved));

        Self {
            year,
            day,
            stars,
            part1: RemotePartState {
                solved: part1_solved,
                answer: answer1,
            },
            part2: RemotePartState {
                solved: part2_solved,
                answer: answer2,
            },
            part2_unlocked: part2_present || part1_solved,
            fetched_at: now(),
        }
    }

    pub fn applied_to_puzzle(&self, puzzle: &Puzzle) -> Self {
        Self::from_puzzle(self.year, self.day, puzzle, self.stars)
    }

    pub fn solved_parts(&self) -> Vec<u8> {
        let mut parts = Vec::new();
        if self.part1.solved {
            parts.push(1);
        }
        if self.part2.solved {
            parts.push(2);
        }
        parts
    }

    pub fn path(archive: &Path, year: u32, day: u32) -> PathBuf {
        scrape::puzzle_dir(archive, year, day).join("remote.json")
    }

    pub fn load(archive: &Path, year: u32, day: u32) -> Option<Self> {
        let text = fs::read_to_string(Self::path(archive, year, day)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self, archive: &Path) -> Result<()> {
        let path = Self::path(archive, self.year, self.day);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionStatus {
    Accepted,
    Rejected,
    TooSoon,
    WrongLevel,
    Unknown,
}

impl SubmissionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::TooSoon => "too_soon",
            Self::WrongLevel => "wrong_level",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Submission {
    pub status: SubmissionStatus,
    pub message: String,
    pub wait_seconds: Option<u64>,
}

/// Parse the HTML returned by `POST /{year}/day/{day}/answer`.
pub fn parse_submission(html: &str) -> Submission {
    let doc = Html::parse_document(html);
    let text = doc.root_element().text().collect::<Vec<_>>().join(" ");
    let message = truncate(text.split_whitespace().collect::<Vec<_>>().join(" "), 500);
    let lower = message.to_lowercase();

    let status = if lower.contains("that's the right answer")
        || lower.contains("that’s the right answer")
    {
        SubmissionStatus::Accepted
    } else if lower.contains("too recently") || lower.contains("left to wait") {
        SubmissionStatus::TooSoon
    } else if lower.contains("right level") || lower.contains("already complete") {
        SubmissionStatus::WrongLevel
    } else if lower.contains("not the right answer") {
        SubmissionStatus::Rejected
    } else {
        SubmissionStatus::Unknown
    };

    Submission {
        status,
        message,
        wait_seconds: parse_wait(&lower),
    }
}

/// Stars per day from an authenticated calendar page.
pub fn calendar_stars(year: u32, html: &str) -> BTreeMap<u32, u8> {
    let doc = Html::parse_document(html);
    let selector = Selector::parse("a[href]").expect("valid selector");
    let prefix = format!("/{year}/day/");

    let mut stars = BTreeMap::new();
    for anchor in doc.select(&selector) {
        let Some(href) = anchor.value().attr("href") else {
            continue;
        };
        let Some(rest) = href.strip_prefix(&prefix) else {
            continue;
        };
        let Ok(day) = rest.parse::<u32>() else {
            continue;
        };

        let mut count = 0;
        for class in anchor.value().classes() {
            match class {
                "calendar-verycomplete" => count = 2,
                "calendar-complete" => count = count.max(1),
                _ => {}
            }
        }
        if count > 0 {
            stars.insert(day, count);
        }
    }
    stars
}

fn parse_wait(text: &str) -> Option<u64> {
    let index = text.find("left to wait")?;
    let start = index.saturating_sub(40);
    let start = (start..=index)
        .find(|&offset| text.is_char_boundary(offset))
        .unwrap_or(index);
    let window = &text[start..index];

    let mut minutes = 0;
    let mut seconds = 0;
    for token in window.split_whitespace() {
        let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if let Some(value) = token.strip_suffix('m').and_then(|n| n.parse::<u64>().ok()) {
            minutes = value;
        } else if let Some(value) = token.strip_suffix('s').and_then(|n| n.parse::<u64>().ok()) {
            seconds = value;
        }
    }

    let total = minutes * 60 + seconds;
    (total > 0).then_some(total)
}

fn truncate(text: String, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text;
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push('…');
    out
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Part;

    #[test]
    fn parses_calendar_stars() {
        let html = r#"
            <a aria-label="Day 1, one star" href="/2025/day/1" class="calendar-day1 calendar-complete"></a>
            <a aria-label="Day 2, two stars" href="/2025/day/2" class="calendar-day2 calendar-verycomplete"></a>
            <a aria-label="Day 3" href="/2025/day/3" class="calendar-day3"></a>
            <a aria-label="Day 4, one star" href="/2025/day/4" class="calendar-day4 calendar-complete"></a>
        "#;
        let stars = calendar_stars(2025, html);
        assert_eq!(stars.get(&1), Some(&1));
        assert_eq!(stars.get(&2), Some(&2));
        assert_eq!(stars.get(&3), None);
        assert_eq!(stars.get(&4), Some(&1));
    }

    #[test]
    fn parses_correct_submission() {
        let html = r#"<main><article><p>That's the right answer! You are one gold star closer.</p></article></main>"#;
        let submission = parse_submission(html);
        assert_eq!(submission.status, SubmissionStatus::Accepted);
    }

    #[test]
    fn parses_wrong_submission() {
        let html = r#"<main><article><p>That's not the right answer; your answer is too low.</p></article></main>"#;
        let submission = parse_submission(html);
        assert_eq!(submission.status, SubmissionStatus::Rejected);
    }

    #[test]
    fn parses_cooldown_and_wait() {
        let html = r#"<main><article>
            <p>You gave an answer too recently; you have to wait until you can submit again.</p>
            <p>You have 4m 12s left to wait.</p>
        </article></main>"#;
        let submission = parse_submission(html);
        assert_eq!(submission.status, SubmissionStatus::TooSoon);
        assert_eq!(submission.wait_seconds, Some(252));
    }

    #[test]
    fn cooldown_parsing_respects_utf8_boundaries() {
        let html = "<p>You have 1m 2s left to wait. That’s all.</p>";
        let submission = parse_submission(html);
        assert_eq!(submission.status, SubmissionStatus::TooSoon);
        assert_eq!(submission.wait_seconds, Some(62));
    }

    #[test]
    fn parses_wrong_level() {
        let html = r#"<main><article><p>You don't seem to be solving the right level. Did you already complete it?</p></article></main>"#;
        let submission = parse_submission(html);
        assert_eq!(submission.status, SubmissionStatus::WrongLevel);
    }

    #[test]
    fn derives_state_from_puzzle() {
        let puzzle = Puzzle {
            year: 2025,
            day: 1,
            title: "Secret Entrance".to_string(),
            url: "https://adventofcode.com/2025/day/1".to_string(),
            fetched_at: 0,
            authenticated: true,
            parts: vec![Part {
                number: 2,
                heading: "--- Part Two ---".to_string(),
                markdown: String::new(),
                html: String::new(),
                examples: Vec::new(),
            }],
            answers: vec![Some("1078".to_string()), None],
            input: None,
        };
        let state = RemoteDayState::from_puzzle(2025, 1, &puzzle, 1);
        assert!(state.part1.solved);
        assert_eq!(state.part1.answer.as_deref(), Some("1078"));
        assert!(state.part2_unlocked);
        assert!(!state.part2.solved);
        assert_eq!(state.stars, 1);
        assert_eq!(state.solved_parts(), vec![1]);
    }

    #[test]
    fn state_round_trips() {
        let dir = std::env::temp_dir().join(format!("harness-remote-test-{}", std::process::id()));
        let mut state = RemoteDayState::new(2025, 1);
        state.part1.solved = true;
        state.part1.answer = Some("1078".to_string());
        state.stars = 1;
        state.save(&dir).unwrap();
        let loaded = RemoteDayState::load(&dir, 2025, 1).unwrap();
        assert!(loaded.part1.solved);
        assert_eq!(loaded.part1.answer.as_deref(), Some("1078"));
        let _ = fs::remove_dir_all(dir);
    }
}
