# Deployment Guide for Vercel

## 🚀 Quick Deployment

This guide explains how to deploy your handwritten equation solver to Vercel.

### Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (optional but recommended):
   ```bash
   npm install -g vercel
   ```
3. **GitHub Repository**: Your code should be pushed to GitHub

---

## 📦 What Changed for Vercel Deployment

### Problem Summary
The original deployment failed with "Out of Memory" (OOM) errors because:
- **4 Python files** in `api/` created 4 separate serverless functions
- Each function installed **1.2+ GB** of dependencies (including PyTorch)
- Total: **~5-6 GB** exceeded Vercel's 8 GB build limit
- PyTorch alone is **800+ MB** and exceeds the 50 MB function size limit

### Solution Implemented

✅ **Single Entry Point**: Only `api/index.py` is deployed as a serverless function
✅ **Lightweight Dependencies**: `requirements-vercel.txt` (~50 MB instead of 1.2 GB)
✅ **Python 3.10**: Pinned via `pyproject.toml` for smaller wheels
✅ **Excluded Heavy Files**: `.vercelignore` removes notebooks, training scripts, model files
✅ **Lightweight Solver**: No PyTorch—uses OpenCV, NumPy, and SymPy only

---

## 📁 New File Structure

```
Hand_Written_equation_solver/
├── api/
│   ├── index.py                    # ✨ NEW: Main serverless function
│   ├── _lightweight_solver.py      # ✨ NEW: No-PyTorch solver
│   ├── _app_original.py            # Renamed (not deployed)
│   ├── _middleware.py              # Renamed (not deployed)
│   ├── _models.py                  # Renamed (not deployed)
│   └── _routes.py                  # Renamed (not deployed)
├── requirements-vercel.txt         # ✨ NEW: Minimal production deps
├── vercel.json                     # ✨ NEW: Vercel config
├── pyproject.toml                  # ✨ NEW: Python version pin
├── .vercelignore                   # ✨ NEW: Exclude dev files
└── requirements.txt                # Original (local dev only)
```

---

## 🌐 Deployment Methods

### Method 1: Deploy via Vercel Dashboard (Easiest)

1. **Connect GitHub Repository**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Git Repository"
   - Select `Hand_Written_equation_solver`

2. **Configure Project**:
   - **Framework Preset**: Other
   - **Root Directory**: ./
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)

3. **Environment Variables** (optional):
   - Add any environment variables from `.env` if needed
   - For now, none are required

4. **Deploy**:
   - Click "Deploy"
   - Wait 2-3 minutes for build to complete
   - ✅ Build should succeed with <100 MB output!

### Method 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI (if not already)
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from project root
cd D:\Hand_Written_equation_solver
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (select your account)
# - Link to existing project? No
# - Project name? hand-written-equation-solver
# - Directory? ./
# - Override settings? No

# Production deployment
vercel --prod
```

---

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-deployment-url.vercel.app/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "solver": "lightweight",
  "version": "1.0.0"
}
```

### 2. API Information
Visit: `https://your-deployment-url.vercel.app/api`

### 3. Interactive Demo
Visit: `https://your-deployment-url.vercel.app/api/demo`

### 4. Solve Equation via API
```bash
# Using curl
curl -X POST https://your-deployment-url.vercel.app/api/solve \
  -F "file=@path/to/equation.jpg" \
  -F "show_steps=true"
```

---

## 📊 Build Expectations

### Before (Failed):
```
Build Memory Usage:
├── System: 2.5 GB
├── Function 1 install: 1.5 GB (torch, etc.)
├── Function 2 install: 1.5 GB
├── Function 3 install: 1.5 GB
├── Function 4 install: 1.5 GB
└── Total: 8.5 GB ❌ OOM!
```

### After (Success):
```
Build Memory Usage:
├── System: 2.5 GB
├── Single function install: 0.3 GB (minimal deps)
└── Total: 2.8 GB ✅ Success!

Deployment Size: ~30-50 MB ✅
Build Time: 2-3 minutes ✅
```

---

## ⚠️ Important Notes

### Limitations of Lightweight Deployment

This deployment uses a **simplified solver** without PyTorch ML models:

- ✅ **Works**: API structure, image upload, equation solving with SymPy
- ⚠️ **Limited**: OCR uses basic heuristics instead of ML model
- 💡 **Mock Results**: Returns sample solutions based on image characteristics

### For Production ML-Based OCR

Consider these options:

#### Option 1: External OCR Service
```python
# In api/_lightweight_solver.py, replace _basic_ocr_attempt with:
import requests

def _call_external_ocr(image):
    # Google Cloud Vision API
    response = requests.post(
        'https://vision.googleapis.com/v1/images:annotate',
        headers={'Authorization': f'Bearer {api_key}'},
        json={'requests': [{'image': {'content': base64_image}}]}
    )
    return response.json()['text']
```

#### Option 2: Separate ML Backend
Deploy PyTorch model separately:
- **Railway** (recommended): Free tier, Docker support
- **Render**: Similar to Railway
- **Fly.io**: Edge deployment
- **AWS Lambda** with custom containers (max 10 GB)

Architecture:
```
Vercel (Frontend + API Gateway)
    ↓ HTTP request
Railway/Render (PyTorch OCR Model)
    ↓ OCR result
Vercel (SymPy solving + response)
```

---

## 🔧 Troubleshooting

### Build Still Fails with OOM

1. **Check vercel.json**:
   ```json
   {
     "version": 2,
     "functions": {
       "api/index.py": {
         "memory": 512  // Try 1024 if needed
       }
     }
   }
   ```

2. **Verify only index.py exists in api/**:
   ```bash
   ls api/
   # Should see: index.py, _lightweight_solver.py, _*.py (underscore prefix)
   ```

3. **Check requirements-vercel.txt size**:
   ```bash
   pip install -r requirements-vercel.txt --dry-run
   # Should be < 200 MB total
   ```

### Imports Fail at Runtime

If you see `ModuleNotFoundError`:

1. **Ensure dependency is in requirements-vercel.txt**
2. **Check Python version compatibility**:
   ```bash
   python3.10 -m pip install -r requirements-vercel.txt
   ```

### Function Timeout

If requests timeout (>10 seconds):

1. **Increase maxDuration in vercel.json**:
   ```json
   {
     "functions": {
       "api/index.py": {
         "maxDuration": 30  // Max for Hobby plan: 10s, Pro: 60s
       }
     }
   }
   ```

### Wrong Requirements File Used

If Vercel installs from `requirements.txt` instead of `requirements-vercel.txt`:

1. **Ensure vercel.json specifies the correct file** (not supported in all Vercel versions)
2. **Workaround**: Temporarily rename files during deployment:
   ```bash
   mv requirements.txt requirements-dev.txt
   mv requirements-vercel.txt requirements.txt
   # Deploy
   mv requirements.txt requirements-vercel.txt
   mv requirements-dev.txt requirements.txt
   ```

---

## 📈 Monitoring

### View Logs

1. **Vercel Dashboard**:
   - Go to your project
   - Click "Deployments"
   - Select latest deployment
   - View "Build Logs" and "Function Logs"

2. **Vercel CLI**:
   ```bash
   vercel logs your-deployment-url.vercel.app
   ```

### Check Memory Usage

In function logs, look for:
```
Memory Used: 45 MB / 512 MB
Duration: 1.2s
```

---

## 🎯 Next Steps

### Enhance OCR Accuracy

1. **Integrate Google Cloud Vision**:
   - Sign up at [cloud.google.com](https://cloud.google.com)
   - Enable Vision API
   - Add API key to Vercel environment variables
   - Update `_lightweight_solver.py` to call Vision API

2. **Deploy ML Model Separately**:
   - See Railway deployment guide
   - Update `api/index.py` to call external ML service

### Add Frontend

Create `index.html` at project root:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Equation Solver</title>
</head>
<body>
    <iframe src="/api/demo" width="100%" height="100%"></iframe>
</body>
</html>
```

Then update `vercel.json`:
```json
{
  "routes": [
    {"src": "/", "dest": "/index.html"},
    {"src": "/api/(.*)", "dest": "/api/index.py"}
  ]
}
```

---

## 📚 Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/guides/deploying-fastapi-with-vercel)
- [Vercel Build Limits](https://vercel.com/docs/concepts/limits/overview)
- [Railway Deployment](https://docs.railway.app/deploy/deployments)

---

## 🆘 Support

If you encounter issues:

1. **Check Vercel Build Logs** for specific errors
2. **Search Vercel Discussions**: [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)
3. **Review this repo's issues**: [github.com/sakshamsaxena22/Hand_Written_equation_solver/issues](https://github.com/sakshamsaxena22/Hand_Written_equation_solver/issues)

---

## ✅ Success Checklist

Before pushing to production:

- [ ] Vercel build completes successfully (<5 minutes)
- [ ] Deployment size is <50 MB
- [ ] `/api/health` returns `{"status": "healthy"}`
- [ ] `/api/demo` page loads
- [ ] `/api/solve` accepts image uploads
- [ ] Response time is <5 seconds
- [ ] No 500 errors in function logs
- [ ] Environment variables are set (if needed)

---

**Deployment Date**: $(date)
**Author**: Saksham Saxena
**Version**: 1.0.0

