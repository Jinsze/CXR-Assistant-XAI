# 🚀 Hugging Face Spaces Deployment Guide

This guide will help you deploy CXR-Assistant to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Create one at [huggingface.co](https://huggingface.co/)
2. **Git Installed**: Download from [git-scm.com](https://git-scm.com/)
3. **Git LFS Installed**: Required for large model files
   ```bash
   # Windows (using Git for Windows)
   git lfs install
   
   # Mac
   brew install git-lfs
   git lfs install
   
   # Linux
   sudo apt-get install git-lfs
   git lfs install
   ```

## 🔧 Step 1: Prepare Your Model File

**IMPORTANT**: Copy your trained model to the correct location:

```bash
# Copy the trained model to the project models directory
# From: C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\models\colab_clahe_eff_final.keras
# To: C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai\models\colab_clahe_eff_final.keras

# PowerShell command:
Copy-Item "C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\models\colab_clahe_eff_final.keras" `
  -Destination "C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai\models\colab_clahe_eff_final.keras"
```

**Verify the model file exists**:
```bash
cd C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai
ls models\
# You should see: colab_clahe_eff_final.keras
```

## 🌐 Step 2: Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name**: `CXR-Assistant` (or your preferred name)
3. **License**: MIT
4. **Select SDK**: Streamlit
5. **Space hardware**: CPU Basic (free tier)
6. **Visibility**: Public (or Private if preferred)
7. Click **"Create Space"**

You'll get a repository URL like: `https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant`

## 📦 Step 3: Initialize Local Git Repository

Open PowerShell and navigate to your project:

```powershell
# Navigate to the project directory
cd C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\lung_disease_ai

# Initialize Git repository
git init

# Set up Git LFS for large files
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"

# Add all files
git add .

# Verify what will be committed
git status

# Make initial commit
git commit -m "Initial commit: CXR-Assistant v1.0 - Deep Learning Chest X-ray Analysis"
```

## 🔗 Step 4: Link to Hugging Face Remote

```powershell
# Add Hugging Face Space as remote
# Replace YOUR_USERNAME with your actual Hugging Face username
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant

# Verify remote is set
git remote -v
```

## 🚀 Step 5: Push to Hugging Face

```powershell
# Push to the main branch
git branch -M main
git push -u origin main
```

**Authentication**: When prompted, use your Hugging Face credentials:
- **Username**: Your Hugging Face username
- **Password**: Use a **Hugging Face Access Token** (NOT your password)
  - Get token at: https://huggingface.co/settings/tokens
  - Click "New token" → "Write" access → Copy the token
  - Use this token as your password

## 🔄 Step 6: Wait for Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant`
2. Hugging Face will automatically:
   - Install dependencies from `requirements.txt`
   - Download model file via Git LFS
   - Start the Streamlit app
3. Wait 3-5 minutes for the build to complete
4. Your app will be live! 🎉

## ✅ Step 7: Verify Deployment

1. Open your Space URL
2. Upload a test chest X-ray image
3. Click "Analyze X-Ray"
4. Verify:
   - Model loads successfully
   - Predictions are generated
   - Grad-CAM++ visualizations appear
   - Confidence scores display correctly

## 🔧 Troubleshooting

### Issue: Model file too large
**Solution**: Git LFS should handle this automatically. Verify:
```bash
git lfs ls-files
# Should show: models/colab_clahe_eff_final.keras
```

### Issue: ModuleNotFoundError
**Solution**: Check `requirements.txt` has all dependencies:
- streamlit
- tensorflow
- numpy
- opencv-python-headless (not opencv-python)
- pillow
- matplotlib
- pandas

### Issue: Model not found error
**Solution**: Verify model path is relative in `utils/predict.py`:
```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'colab_clahe_eff_final.keras')
```

### Issue: Build fails on Hugging Face
**Solution**: Check the build logs in your Space. Common fixes:
1. Ensure all files are committed
2. Verify `.gitattributes` is present
3. Check model file is tracked by LFS: `git lfs ls-files`

## 🔄 Updating Your Deployed App

When you make changes:

```powershell
# Make your changes to the code
# ...

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Update: [describe your changes]"

# Push to Hugging Face
git push origin main
```

Hugging Face will automatically rebuild and redeploy your Space.

## 🎨 Customization After Deployment

### Update Space Settings
1. Go to your Space Settings
2. **README**: Edit to add custom description
3. **Hardware**: Upgrade to GPU if needed (paid)
4. **Visibility**: Change public/private
5. **Secrets**: Add environment variables if needed

### Add Custom Domain (Pro feature)
In Space Settings → Custom Domain

## 📊 Monitor Usage

- **Analytics**: View in Space Settings
- **Logs**: Check real-time logs for errors
- **Metrics**: Monitor inference time and usage

## 🔒 Security Best Practices

1. **Never commit** sensitive data or API keys
2. Use **Hugging Face Secrets** for environment variables
3. Keep model files secure with **private Spaces** if needed
4. Regularly update dependencies for security patches

## 📝 Important Files Checklist

Before deploying, ensure these files exist:

- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - With Hugging Face YAML metadata
- ✅ `.gitattributes` - Git LFS configuration
- ✅ `.gitignore` - Ignore unnecessary files
- ✅ `models/colab_clahe_eff_final.keras` - Your trained model
- ✅ `utils/` - All utility modules
- ✅ `assets/` - Sample images (optional)

## 🎯 Quick Command Reference

```powershell
# Clone your Space (if you want to make changes locally later)
git clone https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant

# Pull latest changes
git pull origin main

# Push updates
git add .
git commit -m "Your message"
git push origin main

# Check Git LFS status
git lfs ls-files

# Verify remote
git remote -v
```

## 🤝 Sharing Your Space

Once deployed, share your Space:
- **Direct Link**: `https://huggingface.co/spaces/YOUR_USERNAME/CXR-Assistant`
- **Embed**: Get embed code from Space Settings
- **API**: Use Gradio API for programmatic access (if enabled)

## 📞 Support

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Community Forum**: https://discuss.huggingface.co/
- **Git LFS Docs**: https://git-lfs.github.com/

---

**Good luck with your deployment! 🚀**

If you encounter issues, check the Hugging Face Space logs and ensure all files are correctly committed with Git LFS.

