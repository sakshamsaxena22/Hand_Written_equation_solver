# Azure Deployment Guide - Full ML Project

Complete guide to deploy your Hand-Written Equation Solver with PyTorch to Azure.

## 🎯 Overview

This guide will help you deploy the **full application** with all ML capabilities to Azure App Service using Docker containers.

### Why Azure for This Project?

✅ **Supports PyTorch** - No size limits like Vercel  
✅ **Full ML Stack** - Deploy complete application with all models  
✅ **Flexible Compute** - Scale CPU/memory as needed  
✅ **Container Support** - Docker deployment for consistent environment  
✅ **Free Tier Available** - Can start with F1 (Free) or B1 (Basic) tier  

---

## 📦 What's Included

- **Full PyTorch OCR Model** - Real handwritten text recognition
- **Complete ML Pipeline** - Preprocessing, segmentation, classification
- **SymPy Solver** - Symbolic mathematics engine
- **FastAPI Backend** - High-performance async API
- **Interactive Demo** - Web interface included

---

## 🛠️ Prerequisites

### 1. Install Required Tools

#### Azure CLI
```powershell
# Download and install from:
https://aka.ms/installazurecliwindows

# Or use winget
winget install Microsoft.AzureCLI

# Verify installation
az --version
```

#### Docker Desktop
```powershell
# Download from:
https://www.docker.com/products/docker-desktop/

# Or use winget
winget install Docker.DockerDesktop

# After installation, restart Windows
# Then verify:
docker --version
```

### 2. Azure Account

- Sign up at: https://azure.microsoft.com/free/
- Get **$200 free credit** for 30 days
- Free tier includes:
  - App Service F1 (Free) - Good for testing
  - Container Registry Basic tier

---

## 🚀 Deployment Options

### Option A: Quick Deploy (Azure Portal + GitHub)
**Best for**: First-time users, visual interface  
**Time**: 10-15 minutes  
**ML Models**: ✅ Full support

### Option B: CLI Deploy (Azure CLI + Docker)
**Best for**: Developers, automation  
**Time**: 15-20 minutes  
**ML Models**: ✅ Full support

---

## 📋 Option A: Quick Deploy via Azure Portal

### Step 1: Prepare Your Code

```powershell
# Navigate to project
cd D:\Hand_Written_equation_solver

# Ensure all Azure files are present
ls Dockerfile
ls azure_app.py
ls requirements.txt

# Commit if needed
git add .
git commit -m "Add Azure deployment configuration"
git push origin main
```

### Step 2: Create Azure Resources

1. **Go to Azure Portal**: https://portal.azure.com

2. **Create Container Registry** (ACR):
   - Click "+ Create a resource"
   - Search "Container Registry"
   - Click "Create"
   - **Settings:**
     - Subscription: Your subscription
     - Resource group: Create new `equation-solver-rg`
     - Registry name: `equationsolver` (must be unique)
     - Location: Choose nearest (e.g., East US)
     - SKU: **Basic** ($5/month, includes 10GB)
   - Click "Review + Create" → "Create"
   - Wait 1-2 minutes for deployment

3. **Enable Admin Access** (for ACR):
   - Go to your Container Registry
   - Left menu → "Access keys"
   - Enable "Admin user"
   - Copy **Username** and **Password** (needed later)

### Step 3: Build and Push Docker Image

```powershell
# Login to Azure
az login

# Login to Container Registry
az acr login --name equationsolver

# Build Docker image (takes 10-15 minutes first time)
docker build -t equationsolver.azurecr.io/equation-solver:latest .

# Push to Azure Container Registry
docker push equationsolver.azurecr.io/equation-solver:latest
```

### Step 4: Create App Service

1. **In Azure Portal**:
   - Click "+ Create a resource"
   - Search "Web App"
   - Click "Create"

2. **Configure**:
   - Subscription: Your subscription
   - Resource group: `equation-solver-rg`
   - Name: `equation-solver-app` (becomes equation-solver-app.azurewebsites.net)
   - Publish: **Docker Container**
   - Operating System: **Linux**
   - Region: Same as Container Registry
   - **App Service Plan:**
     - Click "Create new"
     - Name: `equation-solver-plan`
     - Sku: **B1 (Basic)** - $13/month, 1.75 GB RAM
       - For ML models, minimum B1 recommended
       - P1V2 ($80/month, 3.5 GB RAM) for production
   
3. **Docker Configuration**:
   - Options: **Single Container**
   - Image Source: **Azure Container Registry**
   - Registry: `equationsolver`
   - Image: `equation-solver`
   - Tag: `latest`

4. **Review + Create** → **Create**
   - Wait 2-3 minutes for deployment

### Step 5: Configure App Settings

1. **Go to your App Service** (`equation-solver-app`)

2. **Configuration** → **Application settings**:
   - Add these settings:
   ```
   WEBSITES_PORT = 8000
   PYTHON_VERSION = 3.10
   ```

3. **General settings** → **Startup Command**:
   ```bash
   python -m uvicorn azure_app:app --host 0.0.0.0 --port 8000
   ```

4. **Health check** (optional but recommended):
   - Path: `/health`
   - Enabled: Yes

5. Click "Save"

### Step 6: Test Deployment

```powershell
# Your app URL
$APP_URL = "https://equation-solver-app.azurewebsites.net"

# Test health check
curl "$APP_URL/health"

# Expected response:
# {"status":"healthy","solver":"full_ml","ml_models":true,"version":"2.0.0"}
```

Open in browser:
- **Demo**: https://equation-solver-app.azurewebsites.net/demo
- **API Docs**: https://equation-solver-app.azurewebsites.net/docs

---

## 📋 Option B: CLI Deploy (Automated)

### Complete PowerShell Script

```powershell
# ============================================
# Azure Deployment Script
# ============================================

# Configuration
$RESOURCE_GROUP = "equation-solver-rg"
$LOCATION = "eastus"
$ACR_NAME = "equationsolver"
$APP_PLAN = "equation-solver-plan"
$APP_NAME = "equation-solver-app"
$IMAGE_NAME = "equation-solver"

# Step 1: Login to Azure
Write-Host "Logging in to Azure..." -ForegroundColor Green
az login

# Step 2: Create Resource Group
Write-Host "Creating resource group..." -ForegroundColor Green
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 3: Create Container Registry
Write-Host "Creating Azure Container Registry..." -ForegroundColor Green
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true

# Step 4: Build and push Docker image
Write-Host "Building Docker image..." -ForegroundColor Green
az acr build `
    --registry $ACR_NAME `
    --image "$IMAGE_NAME:latest" `
    --file Dockerfile `
    .

# Step 5: Create App Service Plan
Write-Host "Creating App Service Plan..." -ForegroundColor Green
az appservice plan create `
    --name $APP_PLAN `
    --resource-group $RESOURCE_GROUP `
    --is-linux `
    --sku B1

# Step 6: Create Web App
Write-Host "Creating Web App..." -ForegroundColor Green
az webapp create `
    --resource-group $RESOURCE_GROUP `
    --plan $APP_PLAN `
    --name $APP_NAME `
    --deployment-container-image-name "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest"

# Step 7: Configure Web App
Write-Host "Configuring Web App..." -ForegroundColor Green

# Get ACR credentials
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

# Configure container settings
az webapp config container set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-custom-image-name "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest" `
    --docker-registry-server-url "https://$ACR_NAME.azurecr.io" `
    --docker-registry-server-user $ACR_USERNAME `
    --docker-registry-server-password $ACR_PASSWORD

# Configure app settings
az webapp config appsettings set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --settings WEBSITES_PORT=8000 PYTHON_VERSION=3.10

# Enable logging
az webapp log config `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-container-logging filesystem

# Step 8: Restart app
Write-Host "Restarting app..." -ForegroundColor Green
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

# Step 9: Get URL
Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
$APP_URL = az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName -o tsv
Write-Host "Your app is available at: https://$APP_URL" -ForegroundColor Cyan
Write-Host "Demo: https://$APP_URL/demo" -ForegroundColor Cyan
Write-Host "API Docs: https://$APP_URL/docs" -ForegroundColor Cyan

# Test health check
Write-Host "`nTesting health check..." -ForegroundColor Green
Start-Sleep -Seconds 30  # Wait for app to start
curl "https://$APP_URL/health"
```

**Save as**: `deploy_azure.ps1`

**Run**:
```powershell
cd D:\Hand_Written_equation_solver
.\deploy_azure.ps1
```

---

## 🔧 Configuration & Optimization

### Memory Settings

For ML models, configure appropriate memory:

```powershell
# Check current settings
az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP

# Upgrade to P1V2 for better performance
az appservice plan update `
    --name $APP_PLAN `
    --resource-group $RESOURCE_GROUP `
    --sku P1V2
```

### Environment Variables

Add optional environment variables:

```powershell
az webapp config appsettings set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --settings `
        LOG_LEVEL=INFO `
        MAX_WORKERS=2 `
        TIMEOUT=120
```

### Auto-Scaling (Production)

```powershell
# Enable autoscaling
az monitor autoscale create `
    --resource-group $RESOURCE_GROUP `
    --resource $APP_NAME `
    --resource-type Microsoft.Web/serverfarms `
    --name autoscale-rules `
    --min-count 1 `
    --max-count 3 `
    --count 1

# Add CPU-based scale rule
az monitor autoscale rule create `
    --resource-group $RESOURCE_GROUP `
    --autoscale-name autoscale-rules `
    --condition "CpuPercentage > 75 avg 5m" `
    --scale out 1
```

---

## 🧪 Testing Your Deployment

### 1. Health Check

```powershell
curl https://equation-solver-app.azurewebsites.net/health
```

Expected:
```json
{
  "status": "healthy",
  "solver": "full_ml",
  "ml_models": true,
  "version": "2.0.0",
  "platform": "azure"
}
```

### 2. API Info

```powershell
curl https://equation-solver-app.azurewebsites.net/
```

### 3. Solve Equation (with curl)

```powershell
# Test with an image
curl -X POST "https://equation-solver-app.azurewebsites.net/solve" `
  -F "file=@path\to\equation.jpg" `
  -F "show_steps=true"
```

### 4. Interactive Demo

Open in browser:
```
https://equation-solver-app.azurewebsites.net/demo
```

---

## 📊 Monitoring & Logs

### View Live Logs

```powershell
# Stream logs in real-time
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### Download Logs

```powershell
# Download log files
az webapp log download `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --log-file logs.zip
```

### Application Insights (Optional)

Enable advanced monitoring:

```powershell
# Create Application Insights
az monitor app-insights component create `
    --app equation-solver-insights `
    --location $LOCATION `
    --resource-group $RESOURCE_GROUP `
    --application-type web

# Link to Web App
$INSTRUMENTATION_KEY = az monitor app-insights component show `
    --app equation-solver-insights `
    --resource-group $RESOURCE_GROUP `
    --query instrumentationKey -o tsv

az webapp config appsettings set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

---

## 💰 Cost Estimation

### Development/Testing

| Service | Tier | Cost/Month | Notes |
|---------|------|------------|-------|
| Container Registry | Basic | $5 | 10GB storage |
| App Service | B1 Basic | $13 | 1.75 GB RAM |
| **Total** | | **~$18/month** | |

### Production

| Service | Tier | Cost/Month | Notes |
|---------|------|------------|-------|
| Container Registry | Basic | $5 | 10GB storage |
| App Service | P1V2 Premium | $80 | 3.5 GB RAM, better CPU |
| Application Insights | Basic | ~$5-10 | Based on usage |
| **Total** | | **~$90-95/month** | |

**Free Tier Option**:
- F1 (Free) App Service: 1GB RAM (may struggle with PyTorch)
- Not recommended for ML models

---

## 🔄 Updating Your Deployment

### Update Code

```powershell
# 1. Make changes to code
# 2. Commit changes
git add .
git commit -m "Update application"

# 3. Rebuild and push image
az acr build `
    --registry $ACR_NAME `
    --image "$IMAGE_NAME:latest" `
    --file Dockerfile `
    .

# 4. Restart app (pulls latest image)
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### Update Configuration Only

```powershell
# Update environment variables
az webapp config appsettings set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --settings NEW_SETTING=value

# Restart
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP
```

---

## 🐛 Troubleshooting

### App Won't Start

```powershell
# Check logs
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Common issues:
# 1. Port mismatch - ensure WEBSITES_PORT=8000
# 2. Memory limit - upgrade to B1 or higher
# 3. Startup timeout - increase timeout in Portal
```

### Out of Memory

```powershell
# Check memory usage
az webapp list-runtimes --linux

# Upgrade plan
az appservice plan update `
    --name $APP_PLAN `
    --resource-group $RESOURCE_GROUP `
    --sku P1V2
```

### Image Pull Failed

```powershell
# Verify ACR credentials
az webapp config container show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP

# Reset container settings
az webapp config container set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-custom-image-name "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest"
```

### Slow First Request

This is normal - ML models load on first request (~30-60 seconds).

**Solutions**:
1. Enable "Always On" (requires Basic tier or higher)
2. Add warmup endpoint
3. Use Application Insights to monitor cold starts

---

## 🗑️ Cleanup (Delete Resources)

```powershell
# Delete entire resource group (removes everything)
az group delete --name $RESOURCE_GROUP --yes --no-wait

# Or delete individual resources
az webapp delete --name $APP_NAME --resource-group $RESOURCE_GROUP
az acr delete --name $ACR_NAME --resource-group $RESOURCE_GROUP
az appservice plan delete --name $APP_PLAN --resource-group $RESOURCE_GROUP
```

---

## 📚 Additional Resources

- [Azure App Service Docs](https://docs.microsoft.com/azure/app-service/)
- [Azure Container Registry Docs](https://docs.microsoft.com/azure/container-registry/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

---

## ✅ Success Checklist

Before going to production:

- [ ] Health check returns `{"status": "healthy"}`
- [ ] Demo page loads and works
- [ ] Can upload images and get solutions
- [ ] ML models load successfully
- [ ] Logs show no errors
- [ ] Response time <5 seconds
- [ ] Enable "Always On" (Basic tier+)
- [ ] Configure custom domain (optional)
- [ ] Set up Application Insights
- [ ] Configure backup/restore
- [ ] Document API endpoints

---

**Deployment Version**: 2.0.0  
**Last Updated**: 2025-10-29  
**Platform**: Azure App Service with Docker
