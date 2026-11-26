//! Advent of Code solving harness.
//!
//! [`aoc::Aoc`] is the interface the solving loop depends on, [`model::Model`]
//! is the LLM seam, and [`agent::Agent`] drives one through the other.

pub mod address;
pub mod agent;
pub mod aoc;
pub mod model;
pub mod part;
pub mod remote;
pub mod runner;
pub mod tools;
