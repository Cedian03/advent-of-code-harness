use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tracing::{debug, warn};

const BASE_URL: &str = "https://adventofcode.com";
const USER_AGENT: &str = "harness/0.1 (personal Advent of Code archive; one request per second)";

/// HTTP client for adventofcode.com with an on-disk raw cache.
///
/// The cache makes scrapes resumable and keeps repeat runs off AoC's servers.
/// Everything it writes lives under the gitignored data directory.
pub struct Fetcher {
    agent: ureq::Agent,
    cookie: Option<String>,
    delay: Duration,
    last_request: Option<Instant>,
    cache_dir: PathBuf,
    refresh: bool,
}

impl Fetcher {
    pub fn new(cache_dir: PathBuf, cookie: Option<String>, delay: Duration, refresh: bool) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();

        Self {
            agent,
            cookie: normalize_cookie(cookie),
            delay,
            last_request: None,
            cache_dir,
            refresh,
        }
    }

    pub fn authenticated(&self) -> bool {
        self.cookie.is_some()
    }

    pub fn calendar(&mut self, year: u32) -> Result<String> {
        let path = self
            .cache_dir
            .join("raw")
            .join(year.to_string())
            .join("calendar.html");
        // A live event adds a day each night, so calendars expire quickly.
        self.get_cached_with_ttl(
            &format!("{BASE_URL}/{year}"),
            &path,
            Some(Duration::from_secs(3600)),
            false,
        )
    }

    pub fn day(&mut self, year: u32, day: u32) -> Result<String> {
        let path = self
            .cache_dir
            .join("raw")
            .join(year.to_string())
            .join(format!("day-{day:02}.html"));
        self.get_cached(&format!("{BASE_URL}/{year}/day/{day}"), &path)
    }

    /// Fetch a day page even if a copy is cached. Used when a session is added
    /// after the initial public scrape: the cached page has no Part Two.
    pub fn day_fresh(&mut self, year: u32, day: u32) -> Result<String> {
        let path = self
            .cache_dir
            .join("raw")
            .join(year.to_string())
            .join(format!("day-{day:02}.html"));
        self.get_cached_with_ttl(&format!("{BASE_URL}/{year}/day/{day}"), &path, None, true)
    }

    /// Puzzle inputs are personal, so this only works with a session cookie.
    pub fn input(&mut self, year: u32, day: u32) -> Result<Option<String>> {
        if self.cookie.is_none() {
            return Ok(None);
        }
        let path = self
            .cache_dir
            .join("raw")
            .join(year.to_string())
            .join("inputs")
            .join(format!("day-{day:02}.txt"));
        let body = self.get_cached(&format!("{BASE_URL}/{year}/day/{day}/input"), &path)?;
        Ok(Some(body))
    }

    fn get_cached(&mut self, url: &str, cache_path: &Path) -> Result<String> {
        self.get_cached_with_ttl(url, cache_path, None, false)
    }

    fn get_cached_with_ttl(
        &mut self,
        url: &str,
        cache_path: &Path,
        ttl: Option<Duration>,
        force: bool,
    ) -> Result<String> {
        if !force && !self.refresh {
            let fresh = match ttl {
                Some(ttl) => fs::metadata(cache_path)
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|age| age < ttl),
                None => true,
            };
            if fresh
                && let Ok(cached) = fs::read_to_string(cache_path)
                && !cached.trim().is_empty()
            {
                debug!(%url, path = %cache_path.display(), "cache hit");
                return Ok(cached);
            }
        }

        debug!(%url, "fetching");
        let body = self.get(url)?;
        if body.trim().is_empty() {
            bail!("empty response from {url}");
        }
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(cache_path, &body)
            .with_context(|| format!("writing cache file {}", cache_path.display()))?;
        Ok(body)
    }

    fn get(&mut self, url: &str) -> Result<String> {
        let mut last_error = None;

        for attempt in 1..=4u32 {
            self.throttle();
            let mut request = self
                .agent
                .get(url)
                .header("User-Agent", USER_AGENT)
                .header("Accept", "text/html,text/plain");

            if let Some(cookie) = &self.cookie {
                request = request.header("Cookie", cookie);
            }

            match request.call() {
                Ok(mut response) => {
                    let status = response.status().as_u16();
                    let body = response
                        .body_mut()
                        .read_to_string()
                        .with_context(|| format!("reading body of {url}"))?;
                    if status == 200 {
                        debug!(%url, status, bytes = body.len(), "fetched");
                        return Ok(body);
                    }
                    last_error = Some(anyhow::anyhow!("{url} returned HTTP {status}"));
                    if !is_retryable(status) {
                        break;
                    }
                }
                Err(ureq::Error::StatusCode(status)) => {
                    last_error = Some(anyhow::anyhow!("{url} returned HTTP {status}"));
                    if !is_retryable(status) {
                        break;
                    }
                }
                Err(err) => {
                    last_error = Some(anyhow::anyhow!("{url}: {err}"));
                }
            }

            if attempt < 4 {
                let backoff = Duration::from_secs(5 * u64::from(attempt));
                warn!(
                    %url,
                    attempt,
                    error = %last_error.as_ref().unwrap(),
                    "retrying request"
                );
                thread::sleep(backoff);
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("{url}: request failed")))
    }

    fn throttle(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed();
            if elapsed < self.delay {
                thread::sleep(self.delay - elapsed);
            }
        }
        self.last_request = Some(Instant::now());
    }
}

fn is_retryable(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}

fn normalize_cookie(value: Option<String>) -> Option<String> {
    let value = value?.trim().to_string();
    if value.is_empty() {
        return None;
    }
    if value.contains('=') {
        Some(value)
    } else {
        Some(format!("session={value}"))
    }
}
