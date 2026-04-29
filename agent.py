#!/usr/bin/env python3
"""
AI Code Review & Auto-Fix Agent

Features:
- Reads Git diff from current branch / PR
- Sends diff and repository context to OpenAI
- Produces structured review findings
- Produces a unified diff patch for high-confidence fixes
- Applies patch safely
- Runs tests
- Rolls back patch if tests fail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

BLOCKED_PATH_PREFIXES = (
    ".git/",
    ".github/workflows/",
)

BLOCKED_EXACT_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "requirements.txt",
}


@dataclass
class CmdResult:
    code: int
    out: str
    err: str


def run(cmd: list[str], check: bool = False, cwd: Optional[str] = None) -> CmdResult:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
    result = CmdResult(proc.returncode, proc.stdout, proc.stderr)
    if check and result.code != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"STDOUT:\n{result.out}\nSTDERR:\n{result.err}"
        )
    return result


def log(msg: str) -> None:
    print(msg, flush=True)


def require_git_repo() -> None:
    r = run(["git", "rev-parse", "--show-toplevel"])
    if r.code != 0:
        raise RuntimeError("Current directory is not a Git repository.")


def get_diff(base: Optional[str]) -> str:
    if base:
        r = run(["git", "diff", "--binary", f"{base}...HEAD"], check=True)
        return r.out

    staged = run(["git", "diff", "--cached", "--binary"]).out
    unstaged = run(["git", "diff", "--binary"]).out
    return staged + "\n" + unstaged


def get_changed_files(base: Optional[str]) -> list[str]:
    if base:
        r = run(["git", "diff", "--name-only", f"{base}...HEAD"], check=True)
    else:
        r = run(["git", "diff", "--name-only"], check=True)
    return [x.strip() for x in r.out.splitlines() if x.strip()]


def get_repo_tree(max_files: int = 300) -> str:
    r = run(["git", "ls-files"])
    if r.code != 0:
        return ""

    files = []
    for path in r.out.splitlines():
        if should_skip_context_file(path):
            continue
        files.append(path)
        if len(files) >= max_files:
            break

    return "\n".join(files)


def should_skip_context_file(path: str) -> bool:
    skip_parts = [
        "node_modules/",
        "dist/",
        "build/",
        ".venv/",
        "venv/",
        "__pycache__/",
        ".git/",
        "coverage/",
        ".next/",
        ".turbo/",
    ]
    return any(part in path for part in skip_parts)


def read_small_files(paths: list[str], max_file_bytes: int = 12_000, max_total_bytes: int = 60_000) -> str:
    chunks = []
    total = 0

    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        if should_skip_context_file(p):
            continue
        try:
            raw = path.read_bytes()
        except Exception:
            continue

        if len(raw) > max_file_bytes:
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        block = f"\n--- FILE: {p} ---\n{text}\n"
        if total + len(block.encode("utf-8")) > max_total_bytes:
            break
        chunks.append(block)
        total += len(block.encode("utf-8"))

    return "\n".join(chunks)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    if not text.startswith("{"):
        obj = re.search(r"\{.*\}", text, re.S)
        if obj:
            text = obj.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        Path(".agent.raw_response.txt").write_text(text, encoding="utf-8")
        raise RuntimeError(f"Model did not return valid JSON. Raw response saved to .agent.raw_response.txt. Error: {e}")


def build_prompt(diff: str, tree: str, file_context: str, max_patch_lines: int) -> str:
    return f"""
你是一个严谨的自动化代码评审与修复 Agent。

你的任务：
1. 审查 Git diff 中新增或修改的代码。
2. 找出 bug、安全漏洞、测试失败风险、明显性能问题、明显可维护性问题。
3. 只修复高置信度问题；不要大规模重构；不要改变业务意图。
4. 生成可以被 `git apply` 应用的标准 unified diff。
5. patch 总行数不得超过 {max_patch_lines} 行。
6. 如果没有高置信度修复，patch 必须是空字符串。
7. 只输出 JSON，不要 Markdown，不要解释 JSON 之外的内容。

严禁：
- 不要修改锁文件。
- 不要修改 CI workflow。
- 不要引入新依赖，除非 diff 已经需要。
- 不要删除用户代码的大块逻辑。
- 不要生成非 unified diff。
- 不要输出 ```json 代码块。

输出 JSON 格式：
{{
  "summary": "简短中文总结",
  "risk_level": "low|medium|high",
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "file": "文件路径",
      "line_hint": "相关行提示",
      "problem": "问题描述",
      "fix": "建议或已执行修复"
    }}
  ],
  "patch": "标准 unified diff 字符串；没有修复则为空字符串"
}}

仓库文件列表：
{tree}

相关文件内容：
{file_context}

Git diff：
{diff}
""".strip()


def call_openai(prompt: str, model: str) -> dict[str, Any]:
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a senior software engineer and automated code review repair agent. Return strict JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    text = response.output_text
    return extract_json(text)


def save_review(result: dict[str, Any]) -> None:
    Path(".agent.review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_review(result: dict[str, Any]) -> None:
    print("\n=== AI Review Summary ===")
    print(result.get("summary", ""))

    print("\n=== Risk Level ===")
    print(result.get("risk_level", "unknown"))

    findings = result.get("findings", [])
    print("\n=== Findings ===")
    if not findings:
        print("No findings.")
    else:
        for item in findings:
            print(
                f"- [{item.get('severity', 'unknown')}] "
                f"{item.get('file', '')} {item.get('line_hint', '')}: "
                f"{item.get('problem', '')}"
            )
            if item.get("fix"):
                print(f"  Fix: {item.get('fix')}")


def patch_line_count(patch: str) -> int:
    return len([line for line in patch.splitlines() if line.strip()])


def paths_in_patch(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path != "/dev/null":
                paths.add(path)
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for token in parts[2:4]:
                    if token.startswith("a/") or token.startswith("b/"):
                        paths.add(token[2:])
    return paths


def validate_patch_safety(patch: str, max_patch_lines: int, allow_sensitive_files: bool) -> None:
    if not patch.strip():
        return

    count = patch_line_count(patch)
    if count > max_patch_lines:
        raise RuntimeError(f"Patch too large: {count} lines, limit is {max_patch_lines}.")

    touched = paths_in_patch(patch)
    if not touched:
        raise RuntimeError("Patch does not look like a valid unified diff.")

    if allow_sensitive_files:
        return

    for p in touched:
        if p in BLOCKED_EXACT_FILES:
            raise RuntimeError(f"Patch tries to modify blocked sensitive file: {p}")
        if any(p.startswith(prefix) for prefix in BLOCKED_PATH_PREFIXES):
            raise RuntimeError(f"Patch tries to modify blocked sensitive path: {p}")


def apply_patch(patch: str, max_patch_lines: int, allow_sensitive_files: bool) -> bool:
    if not patch.strip():
        log("No patch proposed.")
        return False

    validate_patch_safety(patch, max_patch_lines, allow_sensitive_files)

    patch_path = Path(".agent.patch")
    patch_path.write_text(patch, encoding="utf-8")

    check = run(["git", "apply", "--check", str(patch_path)])
    if check.code != 0:
        raise RuntimeError(f"Patch validation failed:\n{check.err}\n{check.out}")

    run(["git", "apply", str(patch_path)], check=True)
    log("Patch applied.")
    return True


def auto_detect_test_command() -> Optional[str]:
    if Path("pytest.ini").exists() or Path("pyproject.toml").exists() or Path("tests").exists():
        if run(["sh", "-lc", "command -v pytest"]).code == 0:
            return "pytest -q"

    if Path("package.json").exists():
        if Path("pnpm-lock.yaml").exists() and run(["sh", "-lc", "command -v pnpm"]).code == 0:
            return "pnpm test"
        if Path("yarn.lock").exists() and run(["sh", "-lc", "command -v yarn"]).code == 0:
            return "yarn test"
        if run(["sh", "-lc", "command -v npm"]).code == 0:
            return "npm test"

    if Path("go.mod").exists() and run(["sh", "-lc", "command -v go"]).code == 0:
        return "go test ./..."

    if Path("Cargo.toml").exists() and run(["sh", "-lc", "command -v cargo"]).code == 0:
        return "cargo test"

    return None


def run_tests(test_command: Optional[str], skip_tests: bool) -> bool:
    if skip_tests:
        log("Skipping tests by user request.")
        return True

    cmd = test_command or auto_detect_test_command()
    if not cmd:
        log("No test command detected. Treating as success.")
        return True

    log(f"Running tests: {cmd}")
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode == 0


def rollback_patch() -> None:
    patch = Path(".agent.patch")
    if not patch.exists():
        return

    r = run(["git", "apply", "-R", str(patch)])
    if r.code == 0:
        log("Patch rolled back.")
    else:
        log("Rollback failed. Please inspect repository manually.")
        log(r.err)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI code review and auto-fix agent.")
    parser.add_argument("--base", help="Base branch or commit, e.g. origin/main")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test", help="Test command, e.g. 'pytest -q' or 'npm test'")
    parser.add_argument("--dry-run", action="store_true", help="Do not apply patch")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run tests")
    parser.add_argument("--max-patch-lines", type=int, default=500)
    parser.add_argument("--allow-sensitive-files", action="store_true")
    args = parser.parse_args()

    require_git_repo()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    diff = get_diff(args.base)
    if not diff.strip():
        log("No diff found.")
        return 0

    changed_files = get_changed_files(args.base)
    tree = get_repo_tree()
    file_context = read_small_files(changed_files)

    log("Calling AI review agent...")
    prompt = build_prompt(diff, tree, file_context, args.max_patch_lines)
    result = call_openai(prompt, args.model)

    save_review(result)
    print_review(result)

    patch = result.get("patch", "")
    if not isinstance(patch, str):
        raise RuntimeError("Invalid model output: patch must be a string.")

    if args.dry_run:
        print("\n=== Proposed Patch ===")
        print(patch if patch.strip() else "(no patch)")
        return 0

    applied = False
    try:
        applied = apply_patch(
            patch=patch,
            max_patch_lines=args.max_patch_lines,
            allow_sensitive_files=args.allow_sensitive_files,
        )
        if applied:
            ok = run_tests(args.test, args.skip_tests)
            if not ok:
                rollback_patch()
                print("Tests failed. AI patch was reverted.", file=sys.stderr)
                return 2
        log("Done.")
        return 0
    except Exception as e:
        if applied:
            rollback_patch()
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
