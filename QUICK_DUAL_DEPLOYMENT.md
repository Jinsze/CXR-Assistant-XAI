# ⚡ Quick Dual Deployment Reference

## 🎯 One-Time Setup Commands

```powershell
# 1. Rename Hugging Face remote
git remote rename origin hf

# 2. Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 3. Push to GitHub
git push -u origin main

# 4. Commit GitHub Action
git add .github/workflows/sync-to-huggingface.yml
git commit -m "Add auto-sync to Hugging Face"
git push origin main
```

## 📝 Daily Workflow

### With Auto-Sync (Recommended)
```powershell
git add .
git commit -m "Your message"
git push origin main  # ← Only this! Auto-syncs to HF
```

### Manual Push to Both
```powershell
git add .
git commit -m "Your message"
git push origin main  # → GitHub
git push hf main      # → Hugging Face
```

## 🔧 Setup Requirements

1. **Create GitHub repo** at https://github.com/new
2. **Get HF token** at https://huggingface.co/settings/tokens
3. **Add to GitHub Secrets**: Settings → Secrets → Actions → `HF_TOKEN`

## ✅ Verify Setup

```powershell
# Should show both remotes
git remote -v

# Expected output:
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (fetch)
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (push)
# origin  https://github.com/YOUR_USERNAME/REPO_NAME.git (fetch)
# origin  https://github.com/YOUR_USERNAME/REPO_NAME.git (push)
```

## 🔗 Your URLs

- GitHub: `https://github.com/YOUR_USERNAME/REPO_NAME`
- Hugging Face: `https://huggingface.co/spaces/jinsze/CXR-Assistant`

---

**Full guide**: See `GITHUB_SETUP.md`

