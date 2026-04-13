use std::net::IpAddr;

use crate::config;
use crate::error::{ParamsError, Result};

use super::policy::normalize_policy_entry;
use super::types::{log_inspection, InspectionDecision, InspectionReport, RiskLevel};

pub fn inspect_fetch_url(raw: &str) -> Result<(String, InspectionReport)> {
    let cfg = config::load_with_profile()?;
    let url = normalize_url(raw)?;
    let (scheme, host) = parse_scheme_and_host(&url)?;
    let host = normalize_host(&host);
    let mut reasons = Vec::new();
    let mut decision = InspectionDecision::Allow;
    let mut risk = RiskLevel::Low;

    if cfg.safety.enabled && cfg.safety.inspect_network {
        if cfg.safety.block_private_network && is_blocked_network_host(&host) {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push(
                "private, loopback, link-local, or localhost targets are blocked".to_string(),
            );
        } else if !cfg.safety.network_allowlist.is_empty()
            && !host_matches_allowlist(&host, &cfg.safety.network_allowlist)
        {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push("host is not in the configured network allowlist".to_string());
        } else {
            reasons
                .push("Outbound fetch is limited to explicit public http/https URLs".to_string());
        }
    }

    let summary = if matches!(decision, InspectionDecision::Block) {
        format!("Blocked network fetch to {host}")
    } else {
        format!("Fetching {scheme} URL from {host}")
    };
    let report = InspectionReport {
        operation: "fetch_url".to_string(),
        decision,
        risk,
        summary,
        reasons,
        targets: vec![url.clone()],
        segments: Vec::new(),
        network_targets: vec![host],
    };
    log_inspection(&report);
    Ok((url, report))
}

pub fn inspect_provider_request(
    operation: &str,
    base_url: &str,
    payload_chars: usize,
) -> Result<(String, InspectionReport)> {
    let cfg = config::load_with_profile()?;
    let url = normalize_url(base_url)?;
    let (scheme, host) = parse_scheme_and_host(&url)?;
    let host = normalize_host(&host);
    let mut reasons = Vec::new();
    let mut decision = InspectionDecision::Allow;
    let mut risk = if payload_chars > 20_000 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    if cfg.safety.enabled && cfg.safety.inspect_cloud_requests {
        if !cfg.safety.network_allowlist.is_empty()
            && !host_matches_allowlist(&host, &cfg.safety.network_allowlist)
        {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push("provider host is not in the configured network allowlist".to_string());
        } else {
            reasons.push(format!(
                "Outbound provider request inspected before send ({payload_chars} chars)"
            ));
        }
    }

    let summary = if matches!(decision, InspectionDecision::Block) {
        format!("Blocked provider request to {host}")
    } else {
        format!("Sending {scheme} provider request to {host}")
    };
    let report = InspectionReport {
        operation: operation.to_string(),
        decision,
        risk,
        summary,
        reasons,
        targets: vec![url.clone()],
        segments: Vec::new(),
        network_targets: vec![host],
    };
    log_inspection(&report);
    Ok((url, report))
}

pub fn normalize_url(raw: &str) -> Result<String> {
    let url = raw.trim();
    if url.is_empty() {
        return Err(ParamsError::Config("URL cannot be empty".to_string()));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ParamsError::Config(
            "Only absolute http:// or https:// URLs are supported".to_string(),
        ));
    }
    if url.contains(char::is_whitespace) {
        return Err(ParamsError::Config(
            "URL cannot contain whitespace".to_string(),
        ));
    }
    Ok(url.to_string())
}

fn normalize_host(host: &str) -> String {
    host.trim().trim_matches('.').to_ascii_lowercase()
}

pub(crate) fn host_matches_allowlist(host: &str, allowlist: &[String]) -> bool {
    let host = normalize_host(host);
    allowlist.iter().any(|entry| {
        let Some(entry) = normalize_policy_entry(entry) else {
            return false;
        };
        host == entry || host.ends_with(&format!(".{entry}"))
    })
}

fn parse_scheme_and_host(url: &str) -> Result<(String, String)> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| ParamsError::Config("Invalid URL".to_string()))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .ok_or_else(|| ParamsError::Config("Invalid URL".to_string()))?;
    if authority.is_empty() {
        return Err(ParamsError::Config("URL host cannot be empty".to_string()));
    }

    let host = if authority.starts_with('[') {
        authority
            .split(']')
            .next()
            .map(|value| value.trim_start_matches('[').to_string())
            .ok_or_else(|| ParamsError::Config("Invalid IPv6 host".to_string()))?
    } else {
        authority.split(':').next().unwrap_or(authority).to_string()
    };

    if host.is_empty() {
        return Err(ParamsError::Config("URL host cannot be empty".to_string()));
    }

    Ok((scheme.to_string(), host))
}

pub(crate) fn is_blocked_network_host(host: &str) -> bool {
    let lowered = host.to_ascii_lowercase();
    if lowered == "localhost" {
        return true;
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => {
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.octets()[0] == 0
            }
            IpAddr::V6(v6) => v6.is_loopback() || v6.is_unicast_link_local(),
        };
    }

    false
}
