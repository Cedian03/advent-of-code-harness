//! Part selectors.
//!
//! A Part is a filter over one Day, never part of its address.

use std::fmt;

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartSelector {
    One,
    Two,
    /// The default: every Part the Day offers, including one that becomes
    /// available during a solve.
    #[default]
    All,
}

impl PartSelector {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            "all" => Ok(Self::All),
            other => bail!("expected `1`, `2`, or `all`, got `{other}`"),
        }
    }

    pub fn includes(self, part: u8) -> bool {
        match self {
            Self::One => part == 1,
            Self::Two => part == 2,
            Self::All => (1..=2).contains(&part),
        }
    }

    /// `true` when the selector names one explicit Part rather than everything.
    pub fn explicit(self) -> Option<u8> {
        match self {
            Self::One => Some(1),
            Self::Two => Some(2),
            Self::All => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::Two => "2",
            Self::All => "all",
        }
    }
}

impl fmt::Display for PartSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Clap value parser.
pub fn parse_part(value: &str) -> std::result::Result<PartSelector, String> {
    PartSelector::parse(value).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selectors_and_rejects_addresses() {
        assert_eq!(PartSelector::parse("1").unwrap(), PartSelector::One);
        assert_eq!(PartSelector::parse("2").unwrap(), PartSelector::Two);
        assert_eq!(PartSelector::parse("all").unwrap(), PartSelector::All);
        assert!(PartSelector::parse("2025/1").is_err());
        assert!(PartSelector::parse("0").is_err());
    }

    #[test]
    fn all_includes_both_parts() {
        assert!(PartSelector::All.includes(1));
        assert!(PartSelector::All.includes(2));
        assert!(!PartSelector::One.includes(2));
    }
}
