//! Skill Tool for CodeMason
//! Based on src/tools/SkillTool/ (getPrompt description + schema) from Claude
//! Code. Equivalent to codemason/tools/skill_tool.py.
//!
//! A skill is a markdown file with frontmatter (see utils/skills.rs). Invoking
//! a skill with `context: inline` (default) expands its body into the
//! conversation (returned as the tool result); `context: fork` runs the body
//! in a sub-agent.
//!
//! Substrate note: the original lists available skills via per-turn
//! system-reminder attachments; codemason has no attachment pipeline, so the
//! skill listing is appended to the tool description (built dynamically).

use async_trait::async_trait;
use futures::StreamExt;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::tools::base::{find_tool, CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::skills::{discover_skills, get_skill, model_invocable_skills, Skill};

pub const SKILL_TOOL_NAME: &str = "Skill";

/// argumentSubstitution.ts parseArguments: shell-quote split, falling back to
/// a whitespace split if parsing fails.
fn parse_arguments(args: &str) -> Vec<String> {
    if args.trim().is_empty() {
        return Vec::new();
    }
    match shlex::split(args) {
        Some(parts) => parts,
        None => args.split_whitespace().map(|t| t.to_string()).collect(),
    }
}

/// Port of argumentSubstitution.ts substituteArguments (lines 94-145).
///
/// Replaces $name / $ARGUMENTS[n] / $n / $ARGUMENTS placeholders in `content`.
/// Only when NO placeholder was found (and args is non-empty) does it append
/// "\n\nARGUMENTS: {args}".
pub fn substitute_arguments(
    content: &str,
    args: Option<&str>,
    append_if_no_placeholder: bool,
    argument_names: &[String],
) -> String {
    // undefined/null means no args provided - return content unchanged.
    let Some(args) = args else {
        return content.to_string();
    };

    let parsed = parse_arguments(args);
    let original = content.to_string();
    let mut content = content.to_string();

    // Named arguments ($foo, $bar) map to positions: names[i] -> parsed[i].
    // Python matched `\$name(?![\[\w])`; the regex crate has no lookahead, so
    // the blocking char is captured and reinserted — a blocked match consumes
    // one extra char, which can never hide a later `$` (names are literal).
    // The replacement value is inserted literally (the Python used a function
    // repl so arg text like C:\Users\x or \1 is never parsed as group refs).
    for (i, name) in argument_names.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let val = parsed.get(i).map(|s| s.as_str()).unwrap_or("");
        if let Ok(re) = Regex::new(&format!(r"\${}([\[\w])?", regex::escape(name))) {
            content = re
                .replace_all(&content, |caps: &regex::Captures| match caps.get(1) {
                    Some(blocker) => format!("${}{}", name, blocker.as_str()),
                    None => val.to_string(),
                })
                .into_owned();
        }
    }

    // Indexed arguments $ARGUMENTS[0], $ARGUMENTS[1], ...
    static INDEXED_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\$ARGUMENTS\[(\d+)\]").unwrap());
    let indexed = |caps: &regex::Captures| -> String {
        let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
        parsed.get(idx).cloned().unwrap_or_default()
    };
    content = INDEXED_RE.replace_all(&content, indexed).into_owned();

    // Shorthand indexed arguments $0, $1, ... (not followed by word char —
    // same captured-blocker restructure as above; note \w includes digits, so
    // a blocked `$12x` reinserts verbatim exactly like the Python's failed
    // backtrack).
    static SHORTHAND_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\$(\d+)(\w)?").unwrap());
    content = SHORTHAND_RE
        .replace_all(&content, |caps: &regex::Captures| match caps.get(2) {
            Some(blocker) => format!("${}{}", &caps[1], blocker.as_str()),
            None => {
                let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
                parsed.get(idx).cloned().unwrap_or_default()
            }
        })
        .into_owned();

    // Full arguments string.
    content = content.replace("$ARGUMENTS", args);

    // If no placeholder was substituted and args is non-empty, append.
    if content == original && append_if_no_placeholder && !args.is_empty() {
        content = format!("{content}\n\nARGUMENTS: {args}");
    }

    content
}

const BASE_PROMPT: &str = include_str!("prompts/skill_base_prompt.txt");

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(SKILL_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

// Embedded-shell syntax (promptShellExecution.ts): fenced ```! blocks and
// inline !`cmd`. Inline requires start-of-line or preceding whitespace so a
// literal `!` mid-word doesn't match. Python used `(?<=\s)` (lookbehind); the
// regex crate has none, so the leading char is captured and reinserted at
// splice time.
static SKILL_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"```!\s*\n?([\s\S]*?)\n?```").unwrap());
static SKILL_INLINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)(^|\s)!`([^`]+)`").unwrap());

/// Port of executeShellCommandsInPrompt: run ```! blocks and !`inline`
/// commands via Bash and splice their output into the skill body. Permission
/// is gated through can_use_tool (fail-closed: no callback or a deny leaves
/// the marker untouched, so a skill file can't silently run shell without
/// approval).
async fn execute_shell_commands_in_prompt(
    text: &str,
    context: &ToolUseContext,
    can_use_tool: Option<&CanUseToolFn>,
) -> String {
    use crate::utils::tool_result_storage::process_tool_result_content;

    // (full match text, leading prefix to keep, command)
    let mut matches: Vec<(String, String, String)> = Vec::new();
    for m in SKILL_BLOCK_RE.captures_iter(text) {
