# AI Code Review & Auto-Fix Agent

一个可上传到 GitHub 的自动化代码评审与修复 Agent。

## 功能

- 读取 Pull Request 或本地 Git diff
- 使用 OpenAI 模型做代码评审
- 输出结构化 findings
- 自动生成 unified diff patch
- 安全校验 patch
- 应用修复
- 运行测试
- 测试失败自动回滚
- GitHub Actions 中自动提交修复

## 本地运行

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_key"

# 只评审，不改代码
python agent.py --base origin/main --dry-run

# 自动修复并运行测试
python agent.py --base origin/main --test "pytest -q"
```

## GitHub 使用

1. 上传本项目文件到你的仓库根目录。
2. 在 GitHub 仓库设置中添加 Secret：

```txt
OPENAI_API_KEY
```

3. 提交后，Pull Request 会自动触发 AI Review Fix workflow。

## 常用命令

```bash
python agent.py --base origin/main --dry-run
python agent.py --base origin/main --test "npm test"
python agent.py --base origin/main --test "pytest -q"
python agent.py --base origin/main --max-patch-lines 300
```

## 安全限制

默认会阻止 Agent 修改以下敏感路径：

- `.github/workflows`
- `.env`
- `package-lock.json`
- `pnpm-lock.yaml`
- `yarn.lock`
- `poetry.lock`
- `requirements.txt`

你可以用参数调整：

```bash
python agent.py --allow-sensitive-files
```
