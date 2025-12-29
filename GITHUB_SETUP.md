# 🐙 GitHub + 🤗 Hugging Face Dual Deployment Guide

This guide shows you how to maintain your CXR-Assistant project on both GitHub and Hugging Face Spaces simultaneously.

## 📋 Overview

**Current Setup:**
- ✅ Deployed to Hugging Face Spaces: https://huggingface.co/spaces/jinsze/CXR-Assistant
- ✅ Git repository initialized locally
- ❌ Not yet on GitHub

**Goal:**
- 🎯 Push code to GitHub (for version control, portfolio, collaboration)
- 🎯 Automatically sync to Hugging Face (for live deployment)
- 🎯 Single push command updates both platforms

## 🚀 Quick Setup (5 Steps)

### Step 1: Create GitHub Repository

1. Go to: https://github.com/new
2. **Repository name**: `CXR-Assistant` (or your preferred name)
3. **Description**: "Deep Learning System for Chest X-ray Analysis with Explainable AI"
4. **Visibility**: Public (recommended for portfolio) or Private
5. ⚠️ **DO NOT** initialize with README, .gitignore, or license (we have these)
6. Click **"Create repository"**

### Step 2: Rename Hugging Face Remote

```powershell
cd C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai

# Rename current "origin" (Hugging Face) to "hf"
git remote rename origin hf

# Verify
git remote -v
# Should show:
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (fetch)
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (push)
```

### Step 3: Add GitHub as Origin

```powershell
# Replace YOUR_GITHUB_USERNAME with your actual GitHub username
# Replace REPO_NAME with your repository name (e.g., CXR-Assistant)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git

# Verify both remotes
git remote -v
# Should show:
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (fetch)
# hf      https://huggingface.co/spaces/jinsze/CXR-Assistant (push)
# origin  https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git (fetch)
# origin  https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git (push)
```

### Step 4: Push to GitHub

```powershell
# Push to GitHub (first time)
git push -u origin main

# You'll be prompted for GitHub credentials:
# - Username: Your GitHub username
# - Password: Use a Personal Access Token (NOT your GitHub password)
#   Get token at: https://github.com/settings/tokens
#   Required permissions: repo (full control)
```

### Step 5: Setup Automatic Sync (GitHub → Hugging Face)

#### 5.1: Get Hugging Face Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. **Name**: `GitHub Actions Sync`
4. **Role**: Write
5. Click **"Generate a token"**
6. **Copy the token** (you won't see it again!)

#### 5.2: Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. **Name**: `HF_TOKEN`
5. **Value**: Paste your Hugging Face token
6. Click **"Add secret"**

#### 5.3: Commit and Push GitHub Action

```powershell
# The GitHub Action file is already created at:
# .github/workflows/sync-to-huggingface.yml

# Commit and push
git add .github/workflows/sync-to-huggingface.yml
git commit -m "Add GitHub Action to auto-sync with Hugging Face"
git push origin main
```

## ✅ Done! Now You Can:

### Option 1: Push to Both Manually

```powershell
# Make your changes...
git add .
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Push to Hugging Face
git push hf main
```

### Option 2: Push to GitHub Only (Auto-Sync Enabled)

```powershell
# Make your changes...
git add .
git commit -m "Your commit message"

# Push to GitHub (automatically syncs to Hugging Face)
git push origin main
```

That's it! The GitHub Action will automatically push to Hugging Face within 1-2 minutes.

## 📊 Workflow Diagram

```
Your Local Machine
       │
       │ git push origin main
       ↓
   GitHub Repository
       │
       │ (GitHub Action triggers)
       ↓
Hugging Face Spaces (Live App)
```

## 🔍 Verify GitHub Action

After pushing to GitHub:

1. Go to your GitHub repository
2. Click **"Actions"** tab
3. You'll see **"Sync to Hugging Face Spaces"** workflow
4. Click on the latest run to see logs
5. Green checkmark ✅ = Success!

## 🛠️ Common Commands Reference

### Check remotes
```powershell
git remote -v
```

### Push to specific remote
```powershell
git push origin main    # Push to GitHub
git push hf main        # Push to Hugging Face
```

### Push to both remotes at once
```powershell
git push origin main && git push hf main
```

### Pull from specific remote
```powershell
git pull origin main    # Pull from GitHub
git pull hf main        # Pull from Hugging Face
```

### Update remote URLs (if needed)
```powershell
# Update GitHub remote
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Update Hugging Face remote
git remote set-url hf https://huggingface.co/spaces/jinsze/CXR-Assistant
```

## 🎯 Recommended Workflow

### Daily Development:

1. **Make changes** locally
2. **Test** with `streamlit run app.py`
3. **Commit**: `git add . && git commit -m "Description"`
4. **Push to GitHub**: `git push origin main`
5. **Auto-sync** to Hugging Face happens automatically
6. **Verify** live app at Hugging Face URL

### Manual Hugging Face Push (if needed):

If auto-sync fails or you need immediate update:

```powershell
git push hf main
```

## 🚨 Troubleshooting

### Issue: GitHub push requires username/password

**Solution**: Use a Personal Access Token instead of password

1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Use token as password when pushing

**Better:** Set up credential caching:
```powershell
git config --global credential.helper manager-core
```

### Issue: GitHub Action fails with "Authentication failed"

**Solution**: Check your `HF_TOKEN` secret

1. Go to GitHub repo → Settings → Secrets → Actions
2. Verify `HF_TOKEN` exists
3. If not, add it with your Hugging Face token
4. Re-run the failed Action

### Issue: LFS files not syncing

**Solution**: Ensure Git LFS is installed and tracked

```powershell
git lfs install
git lfs track "*.keras"
git add .gitattributes
git commit -m "Update LFS tracking"
git push origin main
```

### Issue: Want to remove automatic sync

**Solution**: Delete or disable the GitHub Action

```powershell
# Option 1: Delete the workflow file
rm .github/workflows/sync-to-huggingface.yml
git add .
git commit -m "Remove auto-sync to Hugging Face"
git push origin main

# Option 2: Go to GitHub → Actions → Disable workflow
```

## 📁 Repository Structure

After setup, your repository will have:

```
.github/
└── workflows/
    └── sync-to-huggingface.yml    ← Auto-sync workflow

.gitattributes                      ← Git LFS config
.gitignore                          ← Ignore patterns
README.md                           ← Project documentation
requirements.txt                    ← Python dependencies
app.py                              ← Main Streamlit app

models/
└── colab_clahe_eff_final.keras    ← Trained model (LFS)

utils/
├── __init__.py
├── preprocessing.py
├── predict.py
└── gradcam.py

assets/
└── sample_xray.png                 ← Sample images
```

## 🌐 Your Project URLs

After setup, you'll have:

- 🐙 **GitHub Repository**: `https://github.com/YOUR_USERNAME/CXR-Assistant`
  - Version control
  - Code review
  - Collaboration
  - Portfolio showcase

- 🤗 **Hugging Face Space**: `https://huggingface.co/spaces/jinsze/CXR-Assistant`
  - Live deployment
  - Interactive demo
  - Public access
  - Automatic rebuilds

## 💡 Tips

1. **Commit messages**: Use clear, descriptive messages
   - Good: `"Fix: Resolve temperature scaling edge case"`
   - Bad: `"update"`

2. **Branch strategy**: Use branches for new features
   ```powershell
   git checkout -b feature/new-model
   # Make changes...
   git push origin feature/new-model
   # Create Pull Request on GitHub
   ```

3. **README badges**: Add badges to your GitHub README
   ```markdown
   [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/jinsze/CXR-Assistant)
   ```

4. **Documentation**: Keep both READMEs in sync (GitHub and HF)

5. **Issues**: Use GitHub Issues for bug tracking and feature requests

## 📖 Next Steps

After setting up dual deployment:

1. ✅ Add project to your GitHub profile
2. ✅ Add GitHub repository link to your resume/CV
3. ✅ Share on LinkedIn/academic networks
4. ✅ Add to your portfolio website
5. ✅ Include in your university project submission

## 🎓 Academic Use

For your Data Science project at Universiti Malaya:

- **GitHub**: Shows your code, commits, development process
- **Hugging Face**: Shows working demo, live deployment
- **Both together**: Demonstrates full software engineering skills

Include both URLs in:
- Project report
- Presentation slides
- Resume/CV
- Academic portfolio

---

## 📞 Support Resources

- **GitHub Docs**: https://docs.github.com/
- **GitHub Actions**: https://docs.github.com/en/actions
- **Hugging Face Spaces**: https://huggingface.co/docs/hub/spaces
- **Git LFS**: https://git-lfs.github.com/

---

**You're all set! Happy coding! 🚀**

Your project is now version-controlled on GitHub and deployed live on Hugging Face.

