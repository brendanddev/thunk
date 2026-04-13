use super::types::{SearchFileHit, SearchLineHit};

pub(crate) fn clip_inline(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }

    let clipped = trimmed
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{}…", clipped.trim_end())
}

pub(crate) fn parse_list_dir_output(output: &str) -> Vec<String> {
    let mut lines = output.lines();
    let _ = lines.next();
    let body = lines.collect::<Vec<_>>().join("\n");
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn parse_read_file_output(output: &str) -> Option<(String, String)> {
    let path = output
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("File: "))?
        .trim()
        .to_string();
    let start = output.find("```\n")?;
    let rest = &output[start + 4..];
    let end = rest.rfind("\n```")?;
    Some((path, rest[..end].to_string()))
}

pub(crate) fn parse_search_output(output: &str) -> Vec<SearchFileHit> {
    let mut files = Vec::new();
    let mut current: Option<SearchFileHit> = None;

    for raw_line in output.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() || line.starts_with("Search results for ") {
            continue;
        }

        if !raw_line.starts_with(' ') && line.ends_with(':') {
            if let Some(file) = current.take() {
                files.push(file);
            }
            current = Some(SearchFileHit {
                path: line.trim_end_matches(':').to_string(),
                hits: Vec::new(),
            });
            continue;
        }

        if let Some(file) = current.as_mut() {
            let trimmed = raw_line.trim_start();
            let Some((line_number, line_content)) = trimmed.split_once(':') else {
                continue;
            };
            let Ok(line_number) = line_number.trim().parse::<usize>() else {
                continue;
            };
            file.hits.push(SearchLineHit {
                line_number,
                line_content: line_content.trim().to_string(),
            });
        }
    }

    if let Some(file) = current {
        files.push(file);
    }

    files
}
