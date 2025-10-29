# Azure Deployment - Quick Start

Get your Hand-Written Equation Solver with full PyTorch ML models deployed to Azure in 30 minutes!

## 🎯 What You'll Deploy

- ✅ **Full PyTorch OCR Models** - Real handwritten text recognition
- ✅ **Complete ML Pipeline** - All preprocessing, segmentation, classification
- ✅ **FastAPI Backend** - High-performance async API
- ✅ **Interactive Web Demo** - Test immediately in browser
- ✅ **Auto-scaling Ready** - Scale as needed

## 📋 Prerequisites (5 minutes)

### 1. Install Azure CLI

```powershell
# Download installer
https://aka.ms/installazurecliwindows

# Or use winget
winget install Microsoft.AzureCLI

# Verify
az --version
```

### 2. Install Docker Desktop

```powershell
# Download from
https://www.docker.com/products/docker-desktop/

# Or use winget
winget install Docker.DockerDesktop

# Restart Windows after installation
# Verify
docker --version
```

### 3. Azure Account

Sign up for free: https://azure.microsoft.com/free/
- Get $200 free credit for 30 days
- No credit card required for free tier

## 🚀 Deploy (3 Commands, 20 minutes)

### Option 1: Automated Script (Easiest)

```powershell
# Navigate to project
cd D:\Hand_Written_equation_solver

# Run deployment script
.\deploy_azure.ps1

# That's it! ☕ Grab coffee while it deploys
```

**What it does:**
1. Creates Azure resources
2. Builds Docker image with PyTorch
3. Pushes to Azure Container Registry
4. Deploys to App Service
5. Configures everything
6. Tests health check

**Time**: ~20 minutes (mostly Docker build)

---

### Option 2: Manual (3 Commands)

```powershell
# 1. Login & create resources (2 minutes)
az login
az group create --name equation-solver-rg --location eastus
az acr create --name equationsolver --resource-group equation-solver-rg --sku Basic --admin-enabled true

# 2. Build & push image (15 minutes)
az acr build --registry equationsolver --image equation-solver:latest --file Dockerfile .

# 3. Deploy (3 minutes)
az appservice plan create --name equation-solver-plan --resource-group equation-solver-rg --is-linux --sku B1
az webapp create --resource-group equation-solver-rg --plan equation-solver-plan --name equation-solver-app --deployment-container-image-name equationsolver.azurecr.io/equation-solver:latest

# Configure
az webapp config appsettings set --name equation-solver-app --resource-group equation-solver-rg --settings WEBSITES_PORT=8000
```

---

## ✅ Verify Deployment (2 minutes)

### 1. Check Health

```powershell
# Wait 30 seconds for startup, then:
curl https://equation-solver-app.azurewebsites.net/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "solver": "full_ml",
  "ml_models": true,
  "version": "2.0.0",
  "platform": "azure"
}
```

### 2. Try the Demo

Open in browser:
```
https://equation-solver-app.azurewebsites.net/demo
```

**You should see:**
- Upload form
- "ML-POWERED (PyTorch)" badge
- Working equation solver

### 3. Test API

```powershell
# Upload an equation image
curl -X POST "https://equation-solver-app.azurewebsites.net/solve" `
  -F "file=@test_images\equation.jpg" `
  -F "show_steps=true"
```

---

## 📊 What You Just Deployed

### Architecture

```
Internet
   ↓
Azure App Service (B1 Basic, 1.75 GB RAM)
   ↓
Docker Container
   ├─ Python 3.10
   ├─ PyTorch (CPU)
   ├─ FastAPI
   ├─ OpenCV
   └─ Your ML Models
```

### Services Created

| Service | Purpose | Cost/Month |
|---------|---------|------------|
| Azure Container Registry | Stores Docker image | $5 |
| App Service (B1) | Runs application | $13 |
| **Total** | | **$18** |

---

## 🔧 Common Customizations

### Upgrade Performance

```powershell
# Upgrade to Premium tier (3.5 GB RAM, faster CPU)
az appservice plan update `
  --name equation-solver-plan `
  --resource-group equation-solver-rg `
  --sku P1V2

# Cost: ~$80/month
```

### Enable Auto-Scaling

```powershell
az monitor autoscale create `
  --resource-group equation-solver-rg `
  --resource equation-solver-app `
  --resource-type Microsoft.Web/serverfarms `
  --min-count 1 --max-count 3
```

### Add Custom Domain

```powershell
# Map your domain
az webapp config hostname add `
  --webapp-name equation-solver-app `
  --resource-group equation-solver-rg `
  --hostname yourdomain.com
```

---

## 📱 Access Your App

After deployment, your app is available at:

- **Main App**: `https://equation-solver-app.azurewebsites.net/`
- **Demo**: `https://equation-solver-app.azurewebsites.net/demo`
- **API Docs**: `https://equation-solver-app.azurewebsites.net/docs`
- **Health**: `https://equation-solver-app.azurewebsites.net/health`

---

## 🐛 Troubleshooting

### App Not Starting?

```powershell
# Check logs
az webapp log tail --name equation-solver-app --resource-group equation-solver-rg
```

### Out of Memory?

Upgrade to P1V2:
```powershell
az appservice plan update --name equation-solver-plan --resource-group equation-solver-rg --sku P1V2
```

### Slow First Load?

This is normal - ML models load on first request (~30-60 seconds).

**Fix**: Enable "Always On"
```powershell
az webapp config set --name equation-solver-app --resource-group equation-solver-rg --always-on true
```

---

## 📈 Monitoring

### View Real-time Logs

```powershell
az webapp log tail --name equation-solver-app --resource-group equation-solver-rg
```

### Download Logs

```powershell
az webapp log download --name equation-solver-app --resource-group equation-solver-rg --log-file logs.zip
```

### Add Application Insights (Optional)

```powershell
# Create insights
az monitor app-insights component create `
  --app equation-solver-insights `
  --location eastus `
  --resource-group equation-solver-rg

# Link to app
$key = az monitor app-insights component show --app equation-solver-insights --resource-group equation-solver-rg --query instrumentationKey -o tsv
az webapp config appsettings set --name equation-solver-app --resource-group equation-solver-rg --settings APPINSIGHTS_INSTRUMENTATIONKEY=$key
```

---

## 🔄 Update Deployment

### Push Code Changes

```powershell
# 1. Make changes, commit to Git
git add .
git commit -m "Update code"

# 2. Rebuild image
az acr build --registry equationsolver --image equation-solver:latest --file Dockerfile .

# 3. Restart app (pulls latest image)
az webapp restart --name equation-solver-app --resource-group equation-solver-rg
```

**Time**: ~15 minutes (rebuild + restart)

---

## 🗑️ Delete Everything

```powershell
# Delete all resources (stops billing)
az group delete --name equation-solver-rg --yes --no-wait
```

---

## 💰 Cost Control

### Free Tier (Not Recommended for ML)
- F1: 1GB RAM, 60 CPU min/day
- May struggle with PyTorch models

### Development (Recommended)
- B1: 1.75 GB RAM, ~$18/month
- Good for testing & low traffic

### Production
- P1V2: 3.5 GB RAM, ~$95/month
- Best performance for ML workloads

### Stop Billing
Delete resource group when not in use:
```powershell
az group delete --name equation-solver-rg --yes
```

---

## 📚 Full Documentation

- **Complete Guide**: See `AZURE_DEPLOYMENT.md`
- **Troubleshooting**: `AZURE_DEPLOYMENT.md` → Troubleshooting
- **Azure Docs**: https://docs.microsoft.com/azure/app-service/

---

## ✅ Success Checklist

- [ ] Health check returns "healthy"
- [ ] Demo page loads
- [ ] Can upload image and get solution
- [ ] ML solver badge shows "ML-POWERED"
- [ ] API docs accessible at `/docs`
- [ ] Response time <5 seconds

---

## 🎉 You're Done!

Your full ML-powered equation solver is now live on Azure with:

- ✅ PyTorch OCR models
- ✅ Professional FastAPI backend
- ✅ Interactive web interface
- ✅ Auto-scaling ready
- ✅ Production-grade infrastructure

**Cost**: ~$18/month (development) or ~$95/month (production)

**Next Steps:**
1. Test with various equations
2. Share your demo URL
3. Monitor performance in Azure Portal
4. Scale up when ready for production

---

**Questions?** Check `AZURE_DEPLOYMENT.md` for detailed documentation.

**Happy solving! 🧮**
