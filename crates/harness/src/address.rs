//! Day and Event addresses, and the Selections built from them.
//!
//! The harness parses addresses only; spans are the shell's job. A Day is always
//! fully qualified (`YYYY/D`), an Event is `YYYY`, and repeating addresses forms
//! a Selection.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use crate::aoc::TaskId;

/// One address on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Address {
    Day(TaskId),
    /// All 25 Days of one Event.
    Event(u32),
}

impl Address {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() {
            bail!("empty address");
        }

        if let Some((year, day)) = value.split_once('/') {
            if day.contains('-') {
                return Err(range_error(day_range_hint(year, day)));
            }
            let year = parse_year(year, value)?;
            let day = parse_day_number(day, value)?;
            return Ok(Address::Day(TaskId::new(year, day)));
        }

        if value.contains('-') {
            return Err(range_error(year_range_hint(value)));
        }

        let year = parse_year(value, value)?;
        Ok(Address::Event(year))
    }

    pub fn day(self) -> Option<TaskId> {
        match self {
            Address::Day(id) => Some(id),
            Address::Event(_) => None,
        }
    }
}

/// Clap value parser for a single Day address.
pub fn parse_day(value: &str) -> std::result::Result<TaskId, String> {
    match Address::parse(value) {
        Ok(Address::Day(id)) => Ok(id),
        Ok(Address::Event(_)) => Err(format!(
            "`{value}` is an Event; this command takes one Day, e.g. {value}/1"
        )),
        Err(err) => Err(err.to_string()),
    }
}

/// Clap value parser for an optional Event address.
pub fn parse_event(value: &str) -> std::result::Result<u32, String> {
    match Address::parse(value) {
        Ok(Address::Event(year)) => Ok(year),
        Ok(Address::Day(id)) => Err(format!(
            "`{value}` is a Day; this command takes an Event, e.g. {}",
            id.year
        )),
        Err(err) => Err(err.to_string()),
    }
}

/// Clap value parser that only validates an address; Selection resolution
/// happens after parsing so every address can be reported together.
pub fn parse_address(value: &str) -> std::result::Result<String, String> {
    Address::parse(value)
        .map(|_| value.to_string())
        .map_err(|err| err.to_string())
}

/// A list of addresses, resolved to unique Days in ascending order.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    addresses: Vec<Address>,
}

impl Selection {
    pub fn new(addresses: Vec<Address>) -> Self {
        Self { addresses }
    }

    pub fn parse<I, S>(values: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut addresses = Vec::new();
        for value in values {
            addresses.push(Address::parse(value.as_ref())?);
        }
        Ok(Self { addresses })
    }

    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }

    /// The one Day address, when the Selection is exactly one Day. An Event or
    /// several addresses is not a single Day, which `--wait` requires.
    pub fn single_day(&self) -> Option<TaskId> {
        match self.addresses.as_slice() {
            [Address::Day(id)] => Some(*id),
            _ => None,
        }
    }

    /// Every Day denoted by the addresses, deduplicated and ascending.
    ///
    /// An Event contributes all 25 of its Days. Overlaps from shell brace
    /// expansion therefore collapse, and a Day appears once.
    pub fn days(&self) -> Vec<TaskId> {
        let mut days = BTreeSet::new();
        for address in &self.addresses {
            match *address {
                Address::Day(id) => {
                    days.insert(id);
                }
                Address::Event(year) => {
                    for day in 1..=25 {
                        days.insert(TaskId::new(year, day));
                    }
                }
            }
        }
        days.into_iter().collect()
    }

    /// Distinct years in ascending order, preserving no day information.
    pub fn years(&self) -> Vec<u32> {
        self.days()
            .into_iter()
            .map(|id| id.year)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Exact `(year, day)` pairs, or `None` when the selection is empty and the
    /// caller should use its default (every past Event).
    pub fn day_set(&self) -> Option<BTreeSet<(u32, u32)>> {
        if self.is_empty() {
            return None;
        }
        Some(
            self.days()
                .into_iter()
                .map(|id| (id.year, id.day))
                .collect(),
        )
    }
}

fn parse_year(value: &str, whole: &str) -> Result<u32> {
    if !value.is_empty() && value.len() <= 2 && value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!(
            "`{whole}` is not an address; a Day is always written with its year, e.g. 2025/{whole}"
        );
    }
    if value.len() != 4 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("`{whole}` is not an address; a year is four digits, e.g. 2025 or 2025/1");
    }
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid year in `{whole}`"))
}

fn parse_day_number(value: &str, whole: &str) -> Result<u32> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("`{whole}` is not an address; a Day is written YEAR/DAY, e.g. 2025/1");
    }
    let day: u32 = value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid day in `{whole}`"))?;
    if !(1..=25).contains(&day) {
        bail!("day must be 1-25 in `{whole}`");
    }
    Ok(day)
}

fn range_error(hint: String) -> anyhow::Error {
    anyhow::anyhow!(
        "ranges are not addresses; let the shell expand the span, e.g. `harness scrape {hint}`"
    )
}

/// `2015-2025` → `20{15..25}`.
fn year_range_hint(value: &str) -> String {
    match value.split_once('-') {
        Some((from, to)) if is_year(from) && is_year(to) => brace(from, to),
        _ => value.to_string(),
    }
}

/// `2025/1-25` → `2025/{1..25}`.
fn day_range_hint(year: &str, day: &str) -> String {
    match day.split_once('-') {
        Some((from, to)) if is_day(from) && is_day(to) => {
            format!("{year}/{}", brace(from, to))
        }
        _ => format!("{year}/{day}"),
    }
}

fn is_year(value: &str) -> bool {
    value.len() == 4 && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_day(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 2
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && value
            .parse::<u32>()
            .is_ok_and(|day| (1..=25).contains(&day))
}

/// Keep the shared prefix, brace the varying suffix: `2015`, `2025` → `20{15..25}`.
fn brace(from: &str, to: &str) -> String {
    let common = from
        .bytes()
        .zip(to.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    let split = if common == from.len() || common == to.len() {
        common.saturating_sub(1)
    } else {
        common
    };
    format!("{}{{{}..{}}}", &from[..split], &from[split..], &to[split..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_day_and_event_addresses() {
        assert_eq!(
            Address::parse("2025/1").unwrap(),
            Address::Day(TaskId::new(2025, 1))
        );
        assert_eq!(
            Address::parse("2025/01").unwrap(),
            Address::Day(TaskId::new(2025, 1))
        );
        assert_eq!(Address::parse("2025").unwrap(), Address::Event(2025));
    }

    #[test]
    fn bare_day_is_not_an_address() {
        let err = Address::parse("1").unwrap_err().to_string();
        assert!(err.contains("with its year"), "{err}");
    }

    #[test]
    fn year_range_names_the_brace_expansion() {
        let err = Address::parse("2015-2025").unwrap_err().to_string();
        assert!(err.contains("20{15..25}"), "{err}");
    }

    #[test]
    fn day_range_names_the_brace_expansion() {
        let err = Address::parse("2025/1-25").unwrap_err().to_string();
        assert!(err.contains("2025/{1..25}"), "{err}");
    }

    #[test]
    fn selection_collapses_overlaps_and_sorts() {
        let selection = Selection::parse(["2025", "2025/1", "2024/3"]).unwrap();
        let days = selection.days();
        assert_eq!(days.len(), 26);
        assert_eq!(days.first(), Some(&TaskId::new(2024, 3)));
        assert_eq!(days.last(), Some(&TaskId::new(2025, 25)));
        assert_eq!(
            days.iter()
                .filter(|id| **id == TaskId::new(2025, 1))
                .count(),
            1
        );
    }

    #[test]
    fn empty_selection_has_no_default_day_set() {
        assert!(Selection::default().is_empty());
        assert_eq!(Selection::default().day_set(), None);
    }
}
