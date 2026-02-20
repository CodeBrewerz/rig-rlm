//! Restate HTTP helpers for registration and invocation.
//!
//! Ported from finverse-be's restate helper patterns.

use reqwest::Client;
use serde::de::DeserializeOwned;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_ADMIN_PORT: u16 = 9070;
pub const DEFAULT_RESTATE_PORT: u16 = 8080;

/// Log the equivalent curl command for debugging.
fn log_curl_command(label: &str, method: &str, url: &str, body: &str) {
    eprintln!("📡 {label}: curl -X {method} {url} -H 'Content-Type: application/json' -d '{body}'");
}

fn truncate_snippet(text: &str) -> String {
    if text.len() > 200 {
        format!("{}…", &text[..200])
    } else {
        text.to_string()
    }
}

/// Register a service deployment with Restate admin.
pub async fn register_service(
    client: &Client,
    admin_host: &str,
    admin_port: u16,
    service_host: &str,
    service_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("http://{admin_host}:{admin_port}/deployments");
    let body = serde_json::json!({
        "uri": format!("http://{service_host}:{service_port}")
    });
    let body_string = serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string());
    log_curl_command("Register Restate deployment", "POST", &url, &body_string);

    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    eprintln!("Register response status={} body={}", status.as_u16(), text);

    if status.is_success() || status.as_u16() == 409 {
        // 409 = already registered, that's fine
        Ok(())
    } else {
        Err(format!(
            "Failed to register service {service_host}:{service_port} via admin {admin_host}:{admin_port}: {}",
            status
        )
        .into())
    }
}

/// Invoke a Restate workflow by name and key.
pub async fn invoke_workflow<TReq, TResp>(
    client: &Client,
    host: &str,
    port: u16,
    workflow_name: &str,
    workflow_key: &str,
    request: &TReq,
) -> Result<TResp, Box<dyn std::error::Error>>
where
    TReq: serde::Serialize + ?Sized,
    TResp: DeserializeOwned,
{
    let url = format!("http://{host}:{port}/{workflow_name}/{workflow_key}/run");
    if let Ok(body) = serde_json::to_string(request) {
        log_curl_command(&format!("Invoke {workflow_name}"), "POST", &url, &body);
    }

    let resp = client.post(&url).json(request).send().await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    eprintln!(
        "Workflow {} response status={} body_snippet={}",
        workflow_name,
        status.as_u16(),
        truncate_snippet(&text)
    );

    if !status.is_success() {
        return Err(format!("Workflow {} call failed: {}", workflow_name, text).into());
    }

    Ok(serde_json::from_str::<TResp>(&text)?)
}

/// Query workflow status via shared handler.
pub async fn query_workflow_status<TResp>(
    client: &Client,
    host: &str,
    port: u16,
    workflow_name: &str,
    workflow_key: &str,
) -> Result<TResp, Box<dyn std::error::Error>>
where
    TResp: DeserializeOwned,
{
    let url = format!("http://{host}:{port}/{workflow_name}/{workflow_key}/status");
    log_curl_command(&format!("Status {workflow_name}"), "GET", &url, "");

    let resp = client.get(&url).send().await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    eprintln!(
        "Status {} response status={} body={}",
        workflow_name,
        status.as_u16(),
        truncate_snippet(&text)
    );

    if !status.is_success() {
        return Err(format!("Status query failed: {}", text).into());
    }

    Ok(serde_json::from_str::<TResp>(&text)?)
}

pub fn env_host() -> String {
    std::env::var("RESTATE_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string())
}

pub fn env_port(var: &str, default_port: u16) -> u16 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(default_port)
}

pub fn env_admin_port() -> u16 {
    env_port("RESTATE_ADMIN_PORT", DEFAULT_ADMIN_PORT)
}

pub fn env_restate_port() -> u16 {
    env_port("RESTATE_PORT", DEFAULT_RESTATE_PORT)
}
