# `.githooks`

Git hooks for maintaining this repository. Workshop participants do not need them.

- **`pre-commit`:** when a commit includes a marimo notebook, runs it top to bottom with
  `marimo export session` and adds the refreshed `__marimo__/session/<notebook>.py.json` to the same commit.
  These session files are what the molab links show as a static, read-only preview, so the previews always
  match the committed code. The notebooks show their pre-trained models by default, so the previews include
  full results. Commits without notebooks are not affected; a notebook that fails to run stops the commit.

One-time setup (per clone):

```bash
git config core.hooksPath .githooks
```

The hook uses `$MARIMO` if set, otherwise `~/miniconda3/envs/ai-workshop/bin/marimo`, otherwise `marimo` on
the `PATH`. Skip it for one commit with `git commit --no-verify`.
