//! Local archive of Advent of Code puzzles.
//!
//! [`scrape`] downloads and parses day pages into a gitignored cache;
//! [`html`] converts statements to Markdown and extracts examples; [`model`]
//! holds the serialized shapes.

pub mod client;
pub mod html;
pub mod model;
pub mod remote;
pub mod scrape;

pub use model::{DayEntry, Example, Part, Puzzle, RootIndex, YearIndex};
pub use scrape::ScrapeOptions;
