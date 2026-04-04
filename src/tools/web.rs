// src/tools/web.rs
//
// Direct URL fetch tool for pulling explicit web pages into context.

use std::io::Read;

use tracing::info;

use super::{Tool, ToolRunResult};
use crate::error::{ParamsError, Result};
use crate::safety::{self, InspectionDecision};

const MAX_FETCH_BYTES: u64 = 100_000;

pub struct FetchUrlTool;

impl Tool for FetchUrlTool {
    fn name(&self) -> &str {
        "fetch_url"
    }

    fn description(&self) -> &str {
        "Fetch a specific http or https URL as text context. Usage: [fetch_url: https://example.com/docs]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        let (url, inspection) = safety::inspect_fetch_url(arg)?;
        if matches!(inspection.decision, InspectionDecision::Block) {
            return Err(ParamsError::Config(inspection.blocked_message()));
        }
        info!(tool = "fetch_url", "tool called");

        let response = ureq::get(&url)
            .set("User-Agent", "params-cli/0.2")
            .call()
            .map_err(|e| ParamsError::Config(format!("Failed to fetch {url}: {e}")))?;

        let content_type = response
            .header("Content-Type")
            .unwrap_or("text/plain")
            .to_string();

        if !is_textual_content_type(&content_type) {
            return Err(ParamsError::Config(format!(
                "Unsupported content type for {url}: {content_type}"
            )));
        }

        let body = read_response_body(response)?;
        let formatted = format_fetched_content(&url, &content_type, &body);

        Ok(ToolRunResult::Immediate(formatted))
    }
}

fn is_textual_content_type(content_type: &str) -> bool {
    let lowered = content_type.to_ascii_lowercase();
    lowered.starts_with("text/")
        || lowered.starts_with("application/json")
        || lowered.starts_with("application/javascript")
        || lowered.starts_with("application/xml")
        || lowered.starts_with("application/xhtml+xml")
}

fn read_response_body(response: ureq::Response) -> Result<String> {
    let mut reader = response.into_reader().take(MAX_FETCH_BYTES + 1);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let truncated = bytes.len() as u64 > MAX_FETCH_BYTES;
    if truncated {
        bytes.truncate(MAX_FETCH_BYTES as usize);
    }

    let mut text = String::from_utf8_lossy(&bytes).to_string();
    if truncated {
        text.push_str("\n\n[truncated]");
    }

    Ok(text)
}

fn format_fetched_content(url: &str, content_type: &str, body: &str) -> String {
    let content = if content_type.to_ascii_lowercase().contains("html") {
        html_to_text(body)
    } else {
        body.trim().to_string()
    };

    format!("Fetched URL: {url}\nContent-Type: {content_type}\n\n{content}")
}

fn html_to_text(html: &str) -> String {
    let mut output = String::new();
    let mut in_tag = false;
    let mut last_was_whitespace = false;

    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            c => {
                let c = decode_basic_entity(c);
                if c.is_whitespace() {
                    if !last_was_whitespace {
                        output.push(' ');
                        last_was_whitespace = true;
                    }
                } else {
                    output.push(c);
                    last_was_whitespace = false;
                }
            }
        }
    }

    output.trim().to_string()
}

fn decode_basic_entity(ch: char) -> char {
    ch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_http_urls() {
        let result = safety::normalize_url("file:///tmp/test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn accepts_https_urls() {
        let result = safety::normalize_url("https://example.com/docs");
        assert_eq!(result.expect("valid url"), "https://example.com/docs");
    }

    #[test]
    fn recognizes_textual_content_types() {
        assert!(is_textual_content_type("text/html; charset=utf-8"));
        assert!(is_textual_content_type("application/json"));
        assert!(!is_textual_content_type("image/png"));
    }

    #[test]
    fn strips_html_tags_to_text() {
        let text = html_to_text("<html><body><h1>Title</h1><p>Hello world</p></body></html>");
        assert!(text.contains("Title"));
        assert!(text.contains("Hello world"));
        assert!(!text.contains("<h1>"));
    }
}
