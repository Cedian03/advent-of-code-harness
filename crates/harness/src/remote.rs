use std::cell::RefCell;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use aoc_archive::client::Fetcher;
use aoc_archive::html;
use aoc_archive::remote::{RemoteDayState, SubmissionStatus, calendar_stars, parse_submission};
use aoc_archive::scrape::{self, ScrapeOptions};
use tracing::warn;

use crate::aoc::{
    Aoc, ArchiveAoc, RecordReport, RunReport, Submission, SubmitReport, SubmitStatus, Task, TaskId,
    TaskSummary, TestReport, Verdict, append_submission, is_rejected, open_part,
    remove_recorded_answer,
};

/// `Aoc` backed by adventofcode.com, with local evaluation.
///
/// The archive is still the source of statements, inputs, and solution files;
/// the remote side contributes live part/star state and, when `submit` is set,
/// accepts answers. A missing Session only removes personal fetching and
/// submission; public fetching and local solving still work.
pub struct RemoteAoc {
    inner: ArchiveAoc,
    archive: PathBuf,
    cookie: Option<String>,
    delay: Duration,
    fetcher: RefCell<Fetcher>,
}

impl RemoteAoc {
    pub fn new(
        archive: PathBuf,
        solutions: PathBuf,
        build_dir: PathBuf,
        cookie: Option<String>,
        delay: Duration,
    ) -> Self {
        let fetcher = Fetcher::new(archive.clone(), cookie.clone(), delay, false);
        Self {
            inner: ArchiveAoc::new(archive.clone(), solutions, build_dir),
            archive,
            cookie,
            delay,
            fetcher: RefCell::new(fetcher),
        }
    }

    /// Cached state when present, otherwise derived from the archived puzzle.
    pub fn state(&self, id: TaskId) -> Result<RemoteDayState> {
        if let Some(state) = RemoteDayState::load(&self.archive, id.year, id.day) {
            return Ok(state);
        }
        self.state_from_archive(id)
    }

    fn state_from_archive(&self, id: TaskId) -> Result<RemoteDayState> {
        let puzzle = scrape::load_puzzle(&self.archive, id.year, id.day)?;
        let stars = RemoteDayState::load(&self.archive, id.year, id.day)
            .map(|state| state.stars)
            .unwrap_or(0);
        let state = RemoteDayState::from_puzzle(id.year, id.day, &puzzle, stars);
        state.save(&self.archive)?;
        Ok(state)
    }

    /// Reuse a captured Day; fetch only what is absent. A Session also lets a
    /// cached public page be upgraded with the input and Part Two.
    pub fn ensure_day(&self, id: TaskId) -> Result<RemoteDayState> {
        if let Ok(puzzle) = scrape::load_puzzle(&self.archive, id.year, id.day) {
            let complete = puzzle.parts.iter().any(|part| part.number == 2);
            let has_input = crate::aoc::input_path(&self.archive, id).is_file();
            if self.cookie.is_none() || (complete && has_input) {
                return self.state(id);
            }
        }
        scrape::run(&ScrapeOptions {
            years: vec![id.year],
            days: Some([(id.year, id.day)].into_iter().collect()),
            out_dir: self.archive.clone(),
            delay: self.delay,
            refresh: false,
            reparse: false,
            cookie: self.cookie.clone(),
        })?;
        self.state_from_archive(id)
    }

    /// Force a refetch. Only used when we caused the change (our own Submit).
    pub fn refresh_day(&self, id: TaskId) -> Result<RemoteDayState> {
        thread::sleep(self.delay);
        scrape::run(&ScrapeOptions {
            years: vec![id.year],
            days: Some([(id.year, id.day)].into_iter().collect()),
            out_dir: self.archive.clone(),
            delay: self.delay,
            refresh: true,
            reparse: false,
            cookie: self.cookie.clone(),
        })?;
        self.state_from_archive(id)
    }

    /// Read the calendar and update this Day's Star state. Returns whether the
    /// calendar has caught up with the answers we believe were accepted.
    pub fn verify_star(&self, id: TaskId) -> Result<(RemoteDayState, bool)> {
        let calendar_html = {
            let mut fetcher = self.fetcher.borrow_mut();
            fetcher.calendar_fresh(id.year)?
        };
        let stars = calendar_stars(id.year, &calendar_html)
            .get(&id.day)
            .copied()
            .unwrap_or(0);

        let mut state = self.state(id)?;
        let expected = u8::from(state.part1.solved) + u8::from(state.part2.solved);
        let confirmed = expected > 0 && stars >= expected;
        state.stars = state.stars.max(stars);
        state.part1.solved |= stars >= 1;
        state.part2.solved |= stars >= 2;
        if state.part1.solved {
            state.part2_unlocked = true;
        }
        state.save(&self.archive)?;
        Ok((state, confirmed))
    }

    /// Fetch live day + calendar state without touching the archive.
    pub fn fetch_state(&self, id: TaskId) -> Result<RemoteDayState> {
        let authenticated = self.cookie.is_some();
        let day_html = {
            let mut fetcher = self.fetcher.borrow_mut();
            fetcher.day_fresh(id.year, id.day)?
        };
        let calendar_html = {
            let mut fetcher = self.fetcher.borrow_mut();
            fetcher.calendar_fresh(id.year)?
        };
        let puzzle = html::parse_day(id.year, id.day, &day_html, authenticated, None);
        let stars = calendar_stars(id.year, &calendar_html)
            .get(&id.day)
            .copied()
            .unwrap_or(0);
        let state = RemoteDayState::from_puzzle(id.year, id.day, &puzzle, stars);
        state.save(&self.archive)?;
        Ok(state)
    }

    fn submit_remote(&self, id: TaskId, part: u8, answer: &str) -> Result<SubmitReport> {
        if !(1..=2).contains(&part) {
            bail!("part must be 1 or 2");
        }
        if self.cookie.is_none() {
            bail!("--submit needs AOC_SESSION; set it in .env, or drop --submit to record only");
        }

        let answer = answer.trim();
        if answer.is_empty() {
            bail!("answer is empty");
        }

        // A refused Answer is never posted again; the tool layer checks this
        // too, but the client keeps the guarantee even if called directly.
        if is_rejected(&self.archive, id, part, answer) {
            let part2_unlocked = self
                .state(id)
                .map(|state| state.part2_unlocked)
                .unwrap_or(false);
            return Ok(SubmitReport {
                status: SubmitStatus::Rejected,
                verdict: Verdict::Unverified,
                message: "answer was rejected earlier; not resubmitted".to_string(),
                part2_unlocked,
            });
        }

        let mut state = self.state(id)?;
        let solved = if part == 1 {
            state.part1.solved
        } else {
            state.part2.solved
        };
        if solved {
            let known = if part == 1 {
                state.part1.answer.as_deref()
            } else {
                state.part2.answer.as_deref()
            };
            let accepted = known == Some(answer.trim());
            return Ok(SubmitReport {
                status: if accepted {
                    SubmitStatus::Accepted
                } else {
                    SubmitStatus::WrongLevel
                },
                verdict: Verdict::Correct,
                message: format!(
                    "part {part} is already solved{}",
                    known
                        .map(|value| format!(" with `{value}`"))
                        .unwrap_or_default()
                ),
                part2_unlocked: state.part2_unlocked,
            });
        }

        if part == 2 && !state.part2_unlocked {
            return Ok(SubmitReport {
                status: SubmitStatus::WrongLevel,
                verdict: Verdict::Unverified,
                message: "part 2 is not unlocked yet".to_string(),
                part2_unlocked: false,
            });
        }

        let html = {
            let mut fetcher = self.fetcher.borrow_mut();
            fetcher.submit_answer(id.year, id.day, part, answer)?
        };
        let submission = parse_submission(&html);
        let verdict = match submission.status {
            SubmissionStatus::Accepted => Verdict::Correct,
            SubmissionStatus::Rejected => Verdict::Incorrect,
            _ => Verdict::Unverified,
        };

        let entry = Submission {
            at: crate::aoc::now_unix() as u64,
            part,
            answer: answer.to_string(),
            status: submission.status.as_str().to_string(),
            verdict: Some(verdict.as_str().to_string()),
            message: submission.message.clone(),
            wait_seconds: submission.wait_seconds,
        };
        if let Err(err) = append_submission(&self.archive, id, &entry) {
            warn!(%err, "could not append to submissions.jsonl");
        }

        match submission.status {
            SubmissionStatus::Accepted => {
                if part == 1 {
                    state.part1.solved = true;
                    state.part1.answer = Some(answer.to_string());
                    state.part2_unlocked = true;
                } else {
                    state.part2.solved = true;
                    state.part2.answer = Some(answer.to_string());
                }
                state.stars = u8::from(state.part1.solved) + u8::from(state.part2.solved);
                state.save(&self.archive)?;

                if let Err(err) = self.inner.record(id, part, answer) {
                    warn!(%err, "submitted to AoC but could not record the answer locally");
                }

                let mut part2_unlocked = state.part2_unlocked;
                if part == 1 {
                    match self.refresh_day(id) {
                        Ok(refreshed) => {
                            state = refreshed;
                            part2_unlocked = state.part2_unlocked;
                        }
                        Err(err) => {
                            warn!(%err, "submitted part 1 but could not refresh part 2");
                        }
                    }
                }

                Ok(SubmitReport {
                    status: SubmitStatus::Accepted,
                    verdict: Verdict::Correct,
                    message: submission.message,
                    part2_unlocked,
                })
            }
            SubmissionStatus::Rejected => {
                if let Err(err) = remove_recorded_answer(&self.archive, id, part, answer) {
                    warn!(%err, "could not remove the rejected answer from answers.json");
                }
                Ok(SubmitReport {
                    status: SubmitStatus::Rejected,
                    verdict: Verdict::Incorrect,
                    message: submission.message,
                    part2_unlocked: state.part2_unlocked,
                })
            }
            SubmissionStatus::TooSoon => Ok(SubmitReport {
                status: SubmitStatus::TooSoon {
                    wait_seconds: submission.wait_seconds,
                },
                verdict: Verdict::Unverified,
                message: submission.message,
                part2_unlocked: state.part2_unlocked,
            }),
            SubmissionStatus::WrongLevel => Ok(SubmitReport {
                status: SubmitStatus::WrongLevel,
                verdict: Verdict::Unverified,
                message: submission.message,
                part2_unlocked: state.part2_unlocked,
            }),
            SubmissionStatus::Unknown => Ok(SubmitReport {
                status: SubmitStatus::Unknown,
                verdict: Verdict::Unverified,
                message: submission.message,
                part2_unlocked: state.part2_unlocked,
            }),
        }
    }
}

impl Aoc for RemoteAoc {
    fn workspace(&self, id: TaskId) -> Result<PathBuf> {
        self.inner.workspace(id)
    }

    fn tasks(&self) -> Result<Vec<TaskSummary>> {
        self.inner.tasks()
    }

    fn load(&self, id: TaskId) -> Result<Task> {
        self.inner.load(id)
    }

    fn evaluate(&self, id: TaskId, source: &str) -> Result<TestReport> {
        self.inner.evaluate(id, source)
    }

    fn run(&self, id: TaskId, part: u8, source: &str) -> Result<RunReport> {
        self.inner.run(id, part, source)
    }

    fn record(&self, id: TaskId, part: u8, answer: &str) -> Result<RecordReport> {
        self.inner.record(id, part, answer)
    }

    fn open_part(&self, id: TaskId) -> Result<Option<u8>> {
        Ok(open_part(&self.state(id)?))
    }

    fn submit(&self, id: TaskId, part: u8, answer: &str) -> Result<SubmitReport> {
        self.submit_remote(id, part, answer)
    }

    fn verdict(&self, id: TaskId, part: u8, answer: &str) -> Verdict {
        self.inner.verdict(id, part, answer)
    }

    fn is_rejected(&self, id: TaskId, part: u8, answer: &str) -> bool {
        is_rejected(&self.archive, id, part, answer)
    }

    fn submissions(&self, id: TaskId) -> Vec<Submission> {
        crate::aoc::read_submissions(&self.archive, id)
    }
}

pub fn session_cookie_optional() -> Option<String> {
    std::env::var("AOC_SESSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
}

pub fn session_cookie() -> Result<String> {
    session_cookie_optional()
        .context("remote commands require AOC_SESSION (your adventofcode.com session cookie)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aoc_archive::remote::RemotePartState;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("harness-remote-{}-{name}", std::process::id()))
    }

    #[test]
    fn solved_and_locked_parts_do_not_post() {
        let dir = temp_dir("submit");
        let id = TaskId::new(2025, 1);
        let state = RemoteDayState {
            year: 2025,
            day: 1,
            stars: 1,
            part1: RemotePartState {
                solved: true,
                answer: Some("1078".to_string()),
            },
            part2: RemotePartState::default(),
            part2_unlocked: false,
            fetched_at: 0,
        };
        state.save(&dir).unwrap();

        let remote = RemoteAoc::new(
            dir.clone(),
            dir.join("solutions"),
            dir.join("target"),
            Some("cookie".to_string()),
            Duration::from_millis(0),
        );

        let matching = remote.submit(id, 1, "1078").unwrap();
        assert_eq!(matching.status, SubmitStatus::Accepted);

        let different = remote.submit(id, 1, "999").unwrap();
        assert_eq!(different.status, SubmitStatus::WrongLevel);

        let locked = remote.submit(id, 2, "42").unwrap();
        assert_eq!(locked.status, SubmitStatus::WrongLevel);

        let _ = fs::remove_dir_all(dir);
    }
}
