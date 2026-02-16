use super::traits::{Memory, MemoryCategory, MemoryEntry};
use async_trait::async_trait;
use chrono::Local;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::timeout;

/// Lucid Memory component — local-first with distributed context fallback.
///
/// When a local query returns too few results (below `local_hit_threshold`),
/// LucidMemory can fall back to a distributed context provider (the "lucid" binary)
/// to fetch shared knowledge or remote context.
pub struct LucidMemory {
    workspace_dir: PathBuf,
    local: SqliteMemory,
    lucid_cmd: String,
    local_hit_threshold: usize,
    failure_cooldown: Duration,
    sync_timeout: Duration,
    recall_timeout: Duration,
    last_failure_at: Mutex<Option<Instant>>,
}

use super::sqlite::SqliteMemory;

impl LucidMemory {
    pub fn new(workspace_dir: &Path, local: SqliteMemory) -> Self {
        let lucid_cmd = std::env::var("ZEROCLAW_LUCID_CMD").unwrap_or_else(|_| "lucid".to_string());
        let local_hit_threshold = Self::read_env_usize("ZEROCLAW_LUCID_THRESHOLD", 1, 0);

        Self::with_options(
            workspace_dir,
            local,
            lucid_cmd,
            200, // Sync budget
            local_hit_threshold,
            Duration::from_millis(150), // Sync timeout
            Duration::from_millis(800), // Recall timeout
            Duration::from_secs(10),    // Failure cooldown
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        workspace_dir: &Path,
        local: SqliteMemory,
        lucid_cmd: String,
        _sync_budget: usize,
        local_hit_threshold: usize,
        sync_timeout: Duration,
        recall_timeout: Duration,
        failure_cooldown: Duration,
    ) -> Self {
        Self {
            workspace_dir: workspace_dir.to_path_buf(),
            local,
            lucid_cmd,
            local_hit_threshold,
            failure_cooldown,
            sync_timeout,
            recall_timeout,
            last_failure_at: Mutex::new(None),
        }
    }

    fn read_env_usize(name: &str, default: usize, min: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(default, |v| v.max(min))
    }

    fn in_failure_cooldown(&self) -> bool {
        let Ok(guard) = self.last_failure_at.lock() else {
            return false;
        };

        guard
            .as_ref()
            .is_some_and(|last| last.elapsed() < self.failure_cooldown)
    }

    fn mark_failure_now(&self) {
        if let Ok(mut guard) = self.last_failure_at.lock() {
            *guard = Some(Instant::now());
        }
    }

    fn clear_failure(&self) {
        if let Ok(mut guard) = self.last_failure_at.lock() {
            *guard = None;
        }
    }

    fn to_lucid_type(category: &MemoryCategory) -> &'static str {
        match category {
            MemoryCategory::Core => "decision",
            MemoryCategory::Daily => "context",
            MemoryCategory::Conversation => "conversation",
            MemoryCategory::Custom(_) => "learning",
        }
    }

    fn to_memory_category(label: &str) -> MemoryCategory {
        let normalized = label.to_lowercase();
        if normalized.contains("visual") {
            return MemoryCategory::Custom("visual".to_string());
        }

        match normalized.as_str() {
            "decision" | "learning" | "solution" => MemoryCategory::Core,
            "context" | "conversation" => MemoryCategory::Conversation,
            "bug" => MemoryCategory::Daily,
            other => MemoryCategory::Custom(other.to_string()),
        }
    }

    fn merge_results(
        primary_results: Vec<MemoryEntry>,
        secondary_results: Vec<MemoryEntry>,
        limit: usize,
    ) -> Vec<MemoryEntry> {
        if limit == 0 {
            return Vec::new();
        }

        let mut merged = Vec::new();
        let mut seen = HashSet::new();

        for entry in primary_results.into_iter().chain(secondary_results) {
            let signature = format!(
                "{}\u{0}{}",
                entry.key.to_lowercase(),
                entry.content.to_lowercase()
            );

            if seen.insert(signature) {
                merged.push(entry);
                if merged.len() >= limit {
                    break;
                }
            }
        }

        merged
    }

    fn parse_lucid_context(raw: &str) -> Vec<MemoryEntry> {
        let mut in_context_block = false;
        let mut entries = Vec::new();
        let now = Local::now().to_rfc3339();

        for line in raw.lines().map(str::trim) {
            if line == "<lucid-context>" {
                in_context_block = true;
                continue;
            }

            if line == "</lucid-context>" {
                break;
            }

            if !in_context_block || line.is_empty() {
                continue;
            }

            let Some(rest) = line.strip_prefix("- [") else {
                continue;
            };

            let Some((label, content_part)) = rest.split_once(']') else {
                continue;
            };

            let content = content_part.trim();
            if content.is_empty() {
                continue;
            }

            let rank = entries.len();
            entries.push(MemoryEntry {
                id: format!("lucid:{rank}"),
                key: format!("lucid_{rank}"),
                content: content.to_string(),
                category: Self::to_memory_category(label.trim()),
                timestamp: now.clone(),
                session_id: None,
                score: Some((1.0 - rank as f64 * 0.05).max(0.1)),
            });
        }

        entries
    }

    async fn run_lucid_command_raw(
        lucid_cmd: &str,
        args: &[String],
        timeout_window: Duration,
    ) -> anyhow::Result<String> {
        let mut cmd = Command::new(lucid_cmd);
        cmd.args(args);

        let output = timeout(timeout_window, cmd.output()).await.map_err(|_| {
            anyhow::anyhow!(
                "lucid command timed out after {}ms",
                timeout_window.as_millis()
            )
        })??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("lucid command failed: {stderr}");
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn run_lucid_command(
        &self,
        args: &[String],
        timeout: Duration,
    ) -> anyhow::Result<String> {
        Self::run_lucid_command_raw(&self.lucid_cmd, args, timeout).await
    }

    async fn sync_to_lucid_async(&self, key: &str, content: &str, category: &MemoryCategory) {
        let args = vec![
            "store".to_string(),
            format!("{key}: {content}"),
            format!("--type={}", Self::to_lucid_type(category)),
            format!("--project={}", self.workspace_dir.display()),
        ];

        let _ = self.run_lucid_command(&args, self.sync_timeout).await;
    }

    async fn recall_from_lucid(&self, query: &str) -> anyhow::Result<Vec<MemoryEntry>> {
        let args = vec![
            "context".to_string(),
            query.to_string(),
            format!("--budget=200"),
            format!("--project={}", self.workspace_dir.display()),
        ];

        let output = self.run_lucid_command(&args, self.recall_timeout).await?;
        Ok(Self::parse_lucid_context(&output))
    }
}

#[async_trait]
impl Memory for LucidMemory {
    fn name(&self) -> &str {
        "lucid"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
    ) -> anyhow::Result<()> {
        self.local.store(key, content, category.clone()).await?;
        self.sync_to_lucid_async(key, content, &category).await;
        Ok(())
    }

    async fn recall(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryEntry>> {
        let local_results = self.local.recall(query, limit).await?;
        if limit == 0
            || local_results.len() >= limit
            || local_results.len() >= self.local_hit_threshold
        {
            return Ok(local_results);
        }

        if self.in_failure_cooldown() {
            return Ok(local_results);
        }

        match self.recall_from_lucid(query).await {
            Ok(lucid_results) if !lucid_results.is_empty() => {
                self.clear_failure();
                Ok(Self::merge_results(local_results, lucid_results, limit))
            }
            Ok(_) => {
                self.clear_failure();
                Ok(local_results)
            }
            Err(error) => {
                self.mark_failure_now();
                tracing::debug!(
                    command = %self.lucid_cmd,
                    error = %error,
                    "Lucid context unavailable; using local sqlite results"
                );
                Ok(local_results)
            }
        }
    }

    async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>> {
        self.local.get(key).await
    }

    async fn list(&self, category: Option<&MemoryCategory>) -> anyhow::Result<Vec<MemoryEntry>> {
        self.local.list(category).await
    }

    async fn forget(&self, key: &str) -> anyhow::Result<bool> {
        self.local.forget(key).await
    }

    async fn count(&self) -> anyhow::Result<usize> {
        self.local.count().await
    }

    async fn health_check(&self) -> bool {
        self.local.health_check().await
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    fn write_fake_lucid_script(dir: &Path) -> String {
        let script_path = dir.join("fake-lucid.sh");
        let script = r#"#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "store" ]]; then
  echo '{"success":true,"id":"mem_1"}'
  exit 0
fi

if [[ "${1:-}" == "context" ]]; then
  cat <<'EOF'
<lucid-context>
Auth context snapshot
- [decision] Use token refresh middleware
- [context] Working in src/auth.rs
</lucid-context>
EOF
  exit 0
fi

echo "unsupported command" >&2
exit 1
"#;

        fs::write(&script_path, script).unwrap();
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
        script_path.display().to_string()
    }

    fn write_probe_lucid_script(dir: &Path, marker_path: &Path) -> String {
        let script_path = dir.join("probe-lucid.sh");
        let marker = marker_path.display().to_string();
        let script = format!(
            r#"#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1:-}}" == "store" ]]; then
  echo '{{"success":true,"id":"mem_store"}}'
  exit 0
fi

if [[ "${{1:-}}" == "context" ]]; then
  printf 'context\n' >> "{marker}"
  cat <<'EOF'
<lucid-context>
- [decision] should not be used when local hits are enough
</lucid-context>
EOF
  exit 0
fi

echo "unsupported command" >&2
exit 1
"#
        );

        fs::write(&script_path, script).unwrap();
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
        script_path.display().to_string()
    }

    fn test_memory(workspace: &Path, cmd: String) -> LucidMemory {
        let sqlite = SqliteMemory::new(workspace).unwrap();
        LucidMemory::with_options(
            workspace,
            sqlite,
            cmd,
            200,
            3,
            Duration::from_millis(150),
            Duration::from_millis(800),
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn lucid_name() {
        let tmp = TempDir::new().unwrap();
        let memory = test_memory(tmp.path(), "nonexistent-lucid-binary".to_string());
        assert_eq!(memory.name(), "lucid");
    }

    #[tokio::test]
    async fn store_succeeds_when_lucid_missing() {
        let tmp = TempDir::new().unwrap();
        let memory = test_memory(tmp.path(), "nonexistent-lucid-binary".to_string());

        memory
            .store("lang", "User prefers Rust", MemoryCategory::Core)
            .await
            .unwrap();

        let entry = memory.get("lang").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "User prefers Rust");
    }

    #[tokio::test]
    async fn recall_merges_lucid_and_local_results() {
        let tmp = TempDir::new().unwrap();
        let fake_cmd = write_fake_lucid_script(tmp.path());
        let memory = test_memory(tmp.path(), fake_cmd);

        memory
            .store(
                "local_note",
                "Local sqlite auth fallback note",
                MemoryCategory::Core,
            )
            .await
            .unwrap();

        let entries = memory.recall("auth", 5).await.unwrap();

        assert!(entries
            .iter()
            .any(|e| e.content.contains("Local sqlite auth fallback note")));
        assert!(entries.iter().any(|e| e.content.contains("token refresh")));
    }

    #[tokio::test]
    async fn recall_skips_lucid_when_local_hits_are_enough() {
        let tmp = TempDir::new().unwrap();
        let marker = tmp.path().join("context_calls.log");
        let probe_cmd = write_probe_lucid_script(tmp.path(), &marker);

        let sqlite = SqliteMemory::new(tmp.path()).unwrap();
        let memory = LucidMemory::with_options(
            tmp.path(),
            sqlite,
            probe_cmd,
            200,
            1,
            Duration::from_millis(150),
            Duration::from_millis(800),
            Duration::from_secs(10),
        );

        memory
            .store("pref", "Rust should stay local-first", MemoryCategory::Core)
            .await
            .unwrap();

        let entries = memory.recall("rust", 5).await.unwrap();
        assert!(entries
            .iter()
            .any(|e| e.content.contains("Rust should stay local-first")));

        let context_calls = fs::read_to_string(&marker).unwrap_or_default();
        assert!(
            context_calls.trim().is_empty(),
            "Expected local-hit short-circuit; got calls: {context_calls}"
        );
    }

    fn write_failing_lucid_script(dir: &Path, marker_path: &Path) -> String {
        let script_path = dir.join("failing-lucid.sh");
        let marker = marker_path.display().to_string();
        let script = format!(
            r#"#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1:-}}" == "store" ]]; then
  echo '{{"success":true,"id":"mem_store"}}'
  exit 0
fi

if [[ "${{1:-}}" == "context" ]]; then
  printf 'context\n' >> "{marker}"
  echo "simulated lucid failure" >&2
  exit 1
fi

echo "unsupported command" >&2
exit 1
"#
        );

        fs::write(&script_path, script).unwrap();
        let mut perms = fs::metadata(&script_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms).unwrap();
        script_path.display().to_string()
    }

    #[tokio::test]
    async fn failure_cooldown_avoids_repeated_lucid_calls() {
        let tmp = TempDir::new().unwrap();
        let marker = tmp.path().join("failing_context_calls.log");
        let failing_cmd = write_failing_lucid_script(tmp.path(), &marker);

        let sqlite = SqliteMemory::new(tmp.path()).unwrap();
        let memory = LucidMemory::with_options(
            tmp.path(),
            sqlite,
            failing_cmd,
            200,
            99,
            Duration::from_millis(120),
            Duration::from_millis(400),
            Duration::from_secs(5),
        );

        let first = memory.recall("auth", 5).await.unwrap();
        let second = memory.recall("auth", 5).await.unwrap();

        assert!(first.is_empty());
        assert!(second.is_empty());

        let calls = fs::read_to_string(&marker).unwrap_or_default();
        assert_eq!(calls.lines().count(), 1);
    }
}
