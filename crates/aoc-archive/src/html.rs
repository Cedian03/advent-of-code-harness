use ego_tree::NodeRef;
use scraper::{ElementRef, Html, Node, Selector};

use crate::model::{Example, Part, Puzzle, YearIndex};

const BASE_URL: &str = "https://adventofcode.com";

/// Parse a year calendar page, yielding every day that exists for that year.
pub fn parse_calendar(year: u32, html: &str) -> YearIndex {
    let doc = Html::parse_document(html);
    let selector = Selector::parse("a[href]").expect("valid selector");
    let prefix = format!("/{year}/day/");

    let mut days: Vec<u32> = Vec::new();
    for anchor in doc.select(&selector) {
        let Some(href) = anchor.value().attr("href") else {
            continue;
        };
        let Some(rest) = href.strip_prefix(&prefix) else {
            continue;
        };
        if let Ok(day) = rest.parse::<u32>()
            && (1..=25).contains(&day)
            && !days.contains(&day)
        {
            days.push(day);
        }
    }
    days.sort_unstable();

    YearIndex {
        year,
        url: format!("{BASE_URL}/{year}"),
        days: days
            .into_iter()
            .map(|day| crate::model::DayEntry {
                day,
                title: None,
                scraped: false,
                parts_captured: 0,
                tests: 0,
                input: None,
            })
            .collect(),
    }
}

/// Parse a day page into a puzzle. `answers` are filled from the session-only
/// "Your puzzle answer was ..." paragraphs when present.
pub fn parse_day(
    year: u32,
    day: u32,
    html: &str,
    authenticated: bool,
    input: Option<String>,
) -> Puzzle {
    let doc = Html::parse_document(html);
    let article_selector = Selector::parse("article.day-desc").expect("valid selector");

    let articles: Vec<ElementRef<'_>> = doc.select(&article_selector).collect();
    let first_heading = articles
        .first()
        .map(|a| heading_text(*a))
        .unwrap_or_default();
    let title = extract_title(&first_heading).unwrap_or_else(|| format!("Day {day}"));
    let puzzle_url = format!("{BASE_URL}/{year}/day/{day}");

    let parts: Vec<Part> = articles
        .iter()
        .enumerate()
        .map(|(i, article)| {
            let heading = heading_text(*article);
            Part {
                number: (i + 1) as u8,
                html: article.inner_html(),
                markdown: render_markdown(*article, &puzzle_url),
                examples: extract_examples(*article),
                heading,
            }
        })
        .collect();

    let answers = extract_answers(&doc);

    Puzzle {
        year,
        day,
        title,
        url: puzzle_url,
        fetched_at: now_unix(),
        authenticated,
        parts,
        answers,
        input,
    }
}

/// Render a puzzle article (or any subtree) as Markdown.
pub fn render_markdown(el: ElementRef<'_>, base_url: &str) -> String {
    let mut out = String::new();
    render_children(el, &mut out, base_url);
    collapse_excess_blank_lines(&out).trim().to_string()
}

fn render_children(parent: ElementRef<'_>, out: &mut String, base_url: &str) {
    for child in parent.children() {
        render_node(child, out, base_url);
    }
}

fn render_node(node: NodeRef<'_, Node>, out: &mut String, base_url: &str) {
    match node.value() {
        Node::Text(text) => out.push_str(&text.text),
        Node::Element(element) => {
            let Some(el) = ElementRef::wrap(node) else {
                return;
            };
            match element.name() {
                "p" => {
                    block_break(out);
                    render_children(el, out, base_url);
                    out.push_str("\n\n");
                }
                "h1" | "h2" | "h3" | "h4" => {
                    block_break(out);
                    let level = element.name().trim_start_matches('h');
                    out.push_str(&"#".repeat(level.parse().unwrap_or(2)));
                    out.push(' ');
                    out.push_str(&clean_heading(&el.text().collect::<String>()));
                    out.push_str("\n\n");
                }
                "pre" => {
                    block_break(out);
                    let code = el.text().collect::<String>();
                    let code = code.trim_end_matches('\n');
                    out.push_str("```\n");
                    out.push_str(code);
                    out.push_str("\n```\n\n");
                }
                "code" => {
                    let code = el.text().collect::<String>();
                    inline_code(out, &code);
                }
                "em" | "i" => {
                    out.push('*');
                    render_children(el, out, base_url);
                    out.push('*');
                }
                "strong" | "b" => {
                    out.push_str("**");
                    render_children(el, out, base_url);
                    out.push_str("**");
                }
                "ul" => render_list(el, false, out, base_url),
                "ol" => render_list(el, true, out, base_url),
                "li" => {
                    // Only reached for stray list items; the normal path is render_list.
                    render_children(el, out, base_url);
                }
                "br" => out.push('\n'),
                "a" => {
                    let text = el.text().collect::<String>();
                    match el.value().attr("href") {
                        Some(href) => {
                            out.push('[');
                            out.push_str(text.trim());
                            out.push_str("](");
                            out.push_str(&resolve_url(href, base_url));
                            out.push(')');
                        }
                        None => out.push_str(&text),
                    }
                }
                // span, sup, sub, table scaffolding: keep the text, drop styling.
                _ => render_children(el, out, base_url),
            }
        }
        _ => {}
    }
}

fn render_list(el: ElementRef<'_>, ordered: bool, out: &mut String, base_url: &str) {
    block_break(out);
    let mut number = 0;
    for child in el.children() {
        let Some(li) = ElementRef::wrap(child) else {
            continue;
        };
        if li.value().name() != "li" {
            continue;
        }
        number += 1;
        if ordered {
            out.push_str(&format!("{number}. "));
        } else {
            out.push_str("- ");
        }
        let mut item = String::new();
        render_children(li, &mut item, base_url);
        out.push_str(item.trim());
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push('\n');
}

fn resolve_url(href: &str, base_url: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") || href.starts_with("//") {
        return href.to_string();
    }
    if href.starts_with('#') {
        return format!("{base_url}{href}");
    }
    if let Some(path) = href.strip_prefix('/') {
        return format!("{BASE_URL}/{path}");
    }
    // Relative links on AoC resolve against the current day page, e.g.
    // <a href="1"> on /2015/day/25 points at /2015/day/1.
    let base = base_url.trim_end_matches('/');
    let base = base.rsplit_once('/').map(|(dir, _)| dir).unwrap_or(base);
    format!("{base}/{href}")
}

fn inline_code(out: &mut String, code: &str) {
    let longest = longest_backtick_run(code);
    let fence = "`".repeat(if longest == 0 { 1 } else { longest + 1 });
    let pad = if code.starts_with('`') || code.ends_with('`') {
        " "
    } else {
        ""
    };
    out.push_str(&fence);
    out.push_str(pad);
    out.push_str(code);
    out.push_str(pad);
    out.push_str(&fence);
}

fn longest_backtick_run(text: &str) -> usize {
    text.split(|c| c != '`')
        .map(|run| run.len())
        .max()
        .unwrap_or(0)
}

fn block_break(out: &mut String) {
    if out.is_empty() {
        return;
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    if !out.ends_with("\n\n") {
        out.push('\n');
    }
}

fn collapse_excess_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut newlines = 0;
    let mut in_fence = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
        }
        if !in_fence && line.trim().is_empty() {
            newlines += 1;
            if newlines > 1 {
                continue;
            }
        } else {
            newlines = 0;
        }
        out.push_str(line);
    }
    out
}

fn heading_text(article: ElementRef<'_>) -> String {
    let selector = Selector::parse("h2").expect("valid selector");
    article
        .select(&selector)
        .next()
        .map(|h| h.text().collect::<String>())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn clean_heading(text: &str) -> String {
    text.trim().trim_matches('-').trim().to_string()
}

/// "--- Day 1: Historian Hysteria ---" -> "Historian Hysteria".
fn extract_title(heading: &str) -> Option<String> {
    let cleaned = clean_heading(heading);
    let rest = cleaned.strip_prefix("Day ")?;
    let rest = rest.split_once(':').map_or(rest, |(_, title)| title);
    let title = rest.trim();
    (!title.is_empty()).then(|| title.to_string())
}

/// Walk the article in document order and pair each `<pre>` input with the
/// highlighted answer that follows it. AoC marks answers with `<code><em>`.
fn extract_examples(article: ElementRef<'_>) -> Vec<Example> {
    let mut flat = Vec::new();
    flatten(*article, &mut flat);

    let pre_nodes: Vec<usize> = flat
        .iter()
        .enumerate()
        .filter(|(_, n)| is_element(n, "pre"))
        .map(|(i, _)| i)
        .collect();

    let mut examples = Vec::new();
    for (pos, &index) in pre_nodes.iter().enumerate() {
        let Some(pre) = ElementRef::wrap(flat[index]) else {
            continue;
        };
        let input = pre
            .text()
            .collect::<String>()
            .trim_end_matches('\n')
            .to_string();
        if input.is_empty() {
            continue;
        }

        // Only code blocks introduced as examples are treated as examples.
        // AoC also uses <pre> for tables, maps, and rule listings.
        let intro_start = if pos == 0 { 0 } else { pre_nodes[pos - 1] + 1 };
        if !has_example_cue(&flat[intro_start..index]) {
            continue;
        }

        let end = pre_nodes.get(pos + 1).copied().unwrap_or(flat.len());
        let expected = find_answer(&flat[index + 1..end]);
        examples.push(Example { input, expected });
    }
    examples
}

fn has_example_cue(nodes: &[NodeRef<'_, Node>]) -> bool {
    let mut text = String::new();
    for node in nodes {
        if let Node::Text(t) = node.value() {
            text.push_str(&t.text);
        }
    }
    let text = text.to_lowercase();
    [
        "for example",
        "example:",
        "examples:",
        "for instance",
        "suppose",
        "consider",
        "given the following",
        "e.g.",
        "such as",
        "here is",
        "here are",
        "sample",
    ]
    .iter()
    .any(|cue| text.contains(cue))
}

fn find_answer(nodes: &[NodeRef<'_, Node>]) -> Option<String> {
    let mut emphasized = None;
    let mut cued = None;
    let mut trailing_text = String::new();
    for node in nodes {
        let Some(el) = ElementRef::wrap(*node) else {
            if let Node::Text(text) = node.value() {
                trailing_text.push_str(&text.text);
                keep_last_chars(&mut trailing_text, 120);
            }
            continue;
        };
        if el.value().name() != "code" {
            continue;
        }
        let text = el.text().collect::<String>();
        let text = text.trim();
        if text.is_empty() || text.len() > 60 {
            continue;
        }
        let is_emphasized = el
            .select(&Selector::parse("em").expect("valid selector"))
            .next()
            .is_some();
        let cue = trailing_text.to_lowercase();
        let answer_cue = [
            "answer is",
            "answer would be",
            "produces",
            "results in",
            "output is",
            "would give",
        ]
        .iter()
        .any(|needle| cue.contains(needle));
        if is_emphasized {
            // AoC highlights the final answer; the last highlight in the
            // window wins so multi-step examples resolve to the end result.
            emphasized = Some(text.to_string());
        } else if answer_cue {
            cued = Some(text.to_string());
        }
    }
    emphasized.or(cued)
}

fn flatten<'a>(node: NodeRef<'a, Node>, out: &mut Vec<NodeRef<'a, Node>>) {
    for child in node.children() {
        out.push(child);
        flatten(child, out);
    }
}

fn keep_last_chars(text: &mut String, max: usize) {
    let count = text.chars().count();
    if count > max {
        let cut = text
            .char_indices()
            .nth(count - max)
            .map(|(index, _)| index)
            .unwrap_or(0);
        text.drain(..cut);
    }
}

fn is_element(node: &NodeRef<'_, Node>, tag: &str) -> bool {
    matches!(node.value(), Node::Element(el) if el.name() == tag)
}

fn extract_answers(doc: &Html) -> Vec<Option<String>> {
    let selector = Selector::parse("main p").expect("valid selector");
    let code_selector = Selector::parse("code").expect("valid selector");

    let mut answers = Vec::new();
    for p in doc.select(&selector) {
        let text = p.text().collect::<String>();
        if !text.contains("Your puzzle answer was") {
            continue;
        }
        let answer = p
            .select(&code_selector)
            .next()
            .map(|c| c.text().collect::<String>().trim().to_string());
        answers.push(answer);
    }
    answers
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_HTML: &str = r#"
    <html><body><main>
      <article class="day-desc">
        <h2>--- Day 3: Binary Diagnostic ---</h2>
        <p>Some prose with <code>inline</code> and <em>emphasis</em>.</p>
        <p>For example:</p>
        <pre><code>00100
11110
</code></pre>
        <p>The answer would be <code><em>198</em></code>.</p>
        <ul><li>first</li><li>second</li></ul>
      </article>
      <article class="day-desc">
        <h2>--- Part Two ---</h2>
        <p>More text.</p>
        <p>For example:</p>
        <pre><code>abc</code></pre>
        <p>This produces <code>42</code>.</p>
      </article>
    </main></body></html>
    "#;

    #[test]
    fn parses_title_and_parts() {
        let puzzle = parse_day(2021, 3, DAY_HTML, false, None);
        assert_eq!(puzzle.title, "Binary Diagnostic");
        assert_eq!(puzzle.parts.len(), 2);
        assert_eq!(puzzle.parts[0].number, 1);
        assert_eq!(puzzle.parts[1].number, 2);
    }

    #[test]
    fn renders_markdown() {
        let puzzle = parse_day(2021, 3, DAY_HTML, false, None);
        let md = &puzzle.parts[0].markdown;
        assert!(md.contains("`inline`"));
        assert!(md.contains("*emphasis*"));
        assert!(md.contains("```\n00100\n11110\n```"));
        assert!(md.contains("- first\n- second"));
    }

    #[test]
    fn extracts_examples_and_answers() {
        let puzzle = parse_day(2021, 3, DAY_HTML, false, None);
        assert_eq!(puzzle.parts[0].examples.len(), 1);
        assert_eq!(puzzle.parts[0].examples[0].input, "00100\n11110");
        assert_eq!(puzzle.parts[0].examples[0].expected.as_deref(), Some("198"));
        assert_eq!(puzzle.parts[1].examples[0].expected.as_deref(), Some("42"));
    }

    #[test]
    fn parses_calendar_days() {
        let html = r#"<a href="/2025/day/1">1</a><a href="/2025/day/12">12</a>
                      <a href="/2025/about">about</a><a href="/2024/day/3">x</a>"#;
        let index = parse_calendar(2025, html);
        let days: Vec<u32> = index.days.iter().map(|d| d.day).collect();
        assert_eq!(days, vec![1, 12]);
    }

    #[test]
    fn trims_answer_buffer_on_char_boundaries() {
        let mut text = String::from("αβγδε ζηθ");
        keep_last_chars(&mut text, 2);
        assert_eq!(text, "ηθ");
    }

    #[test]
    fn resolves_relative_links_against_the_day_page() {
        let base = "https://adventofcode.com/2015/day/25";
        assert_eq!(
            resolve_url("1", base),
            "https://adventofcode.com/2015/day/1"
        );
        assert_eq!(
            resolve_url("/2015/about", base),
            "https://adventofcode.com/2015/about"
        );
        assert_eq!(resolve_url("#part2", base), format!("{base}#part2"));
        assert_eq!(
            resolve_url("https://example.com", base),
            "https://example.com"
        );
    }
}
