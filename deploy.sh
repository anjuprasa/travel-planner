#!/usr/bin/env bash
# Pushes this project to GitHub so Streamlit Community Cloud can deploy it.
# Usage: ./deploy.sh https://github.com/<your-username>/<repo>.git
set -e
REPO_URL="${1:?Usage: ./deploy.sh <github-repo-url>}"
[ -d .git ] || git init -b main
git add .
git commit -m "Agentic AI Travel Planner" || true
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main
echo ""
echo "Pushed. Final step (30 seconds):"
echo "1. Open https://share.streamlit.io and click 'Create app'"
echo "2. Pick this repo, branch 'main', main file 'app.py'"
echo "3. Advanced settings -> Secrets: paste OPENAI_API_KEY or GOOGLE_API_KEY"
echo "4. Click Deploy"
