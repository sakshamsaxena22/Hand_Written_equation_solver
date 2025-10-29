# Deploy from GitHub to Azure - Complete Guide

Automatically deploy your equation solver from GitHub to Azure App Service with continuous deployment.

## 🎯 Benefits of GitHub Integration

- ✅ **Automatic Deployments** - Push to GitHub → Auto-deploys to Azure
- ✅ **No Local Tools Needed** - No Docker or Azure CLI required on your machine
- ✅ **CI/CD Built-in** - Azure builds Docker image for you
- ✅ **Easy Rollbacks** - Revert to previous versions easily
- ✅ **Free for Public Repos** - GitHub Actions free tier included

---

## 🚀 Method 1: Azure Portal (Easiest - 10 minutes)

This method uses Azure's built-in GitHub integration.

### Prerequisites
- GitHub account with your repo
- Azure account (free tier works)
- Your GitHub repo: `https://github.com/sakshamsaxena22/Hand_Written_equation_solver`

### Step 1: Create Azure Resources

1. **Go to Azure Portal**: https://portal.azure.com

2. **Create Web App**:
   - Click "+ Create a resource"
   - Search "Web App"
   - Click "Create"

3. **Configure Basics**:
   ```
   Subscription: Your subscription
   Resource Group: Create new "equation-solver-rg"
   Name: "equation-solver-app" (or your preferred name)
   Publish: Docker Container
   Operating System: Linux
   Region: East US (or nearest)
   ```

4. **App Service Plan**:
   ```
   Create new plan:
   - Name: equation-solver-plan
   - Sku: B1 Basic ($13/month) - Minimum for PyTorch
   ```

5. Click "Next: Docker"

### Step 2: Configure Docker from GitHub

1. **Docker Settings**:
   ```
   Options: Single Container
   Image Source: GitHub Actions
   ```

2. **GitHub Configuration**:
   - Click "Sign in to GitHub"
   - Authorize Azure
   - Select:
     - Organization: sakshamsaxena22
     - Repository: Hand_Written_equation_solver
     - Branch: main

3. **Dockerfile Settings**:
   ```
   Dockerfile: Dockerfile (in root)
   Startup Command: python -m uvicorn azure_app:app --host 0.0.0.0 --port 8000
   ```

4. Click "Review + Create" → "Create"

### Step 3: Configure Container Registry

Azure will automatically:
1. Create a Container Registry for you
2. Create GitHub Actions workflow
3. Build your Docker image
4. Deploy to App Service

**Wait 5-10 minutes for first deployment**

### Step 4: Configure App Settings

1. Go to your Web App → **Configuration** → **Application settings**

2. Add these settings:
   ```
   WEBSITES_PORT = 8000
   PYTHON_VERSION = 3.10
   ```

3. Click "Save"

4. **Restart** the app

### Step 5: Test Your Deployment

```powershell
# Replace with your actual app name
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

**Try the demo:**
```
https://equation-solver-app.azurewebsites.net/demo
```

---

## 🚀 Method 2: GitHub Actions (More Control)

This method gives you full control over the CI/CD pipeline.

### Step 1: Create GitHub Actions Workflow

I'll create a workflow file for you:

**File**: `.github/workflows/azure-deploy.yml`

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  AZURE_WEBAPP_NAME: equation-solver-app
  AZURE_CONTAINER_REGISTRY: equationsolver
  IMAGE_NAME: equation-solver

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Log in to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Log in to Azure Container Registry
      uses: azure/docker-login@v1
      with:
        login-server: ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io
        username: ${{ secrets.ACR_USERNAME }}
        password: ${{ secrets.ACR_PASSWORD }}
    
    - name: Build and push Docker image
      run: |
        docker build -t ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }} \
                     -t ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io/${{ env.IMAGE_NAME }}:latest .
        docker push ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
        docker push ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io/${{ env.IMAGE_NAME }}:latest
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: ${{ env.AZURE_WEBAPP_NAME }}
        images: ${{ env.AZURE_CONTAINER_REGISTRY }}.azurecr.io/${{ env.IMAGE_NAME }}:${{ github.sha }}
    
    - name: Test deployment
      run: |
        sleep 60
        curl -f https://${{ env.AZURE_WEBAPP_NAME }}.azurewebsites.net/health || exit 1
```

### Step 2: Create Azure Resources (One-time)

```powershell
# Login to Azure
az login

# Create resources
az group create --name equation-solver-rg --location eastus

az acr create \
  --resource-group equation-solver-rg \
  --name equationsolver \
  --sku Basic \
  --admin-enabled true

az appservice plan create \
  --name equation-solver-plan \
  --resource-group equation-solver-rg \
  --is-linux \
  --sku B1

az webapp create \
  --resource-group equation-solver-rg \
  --plan equation-solver-plan \
  --name equation-solver-app \
  --deployment-container-image-name equationsolver.azurecr.io/equation-solver:latest

# Configure app
az webapp config appsettings set \
  --name equation-solver-app \
  --resource-group equation-solver-rg \
  --settings WEBSITES_PORT=8000 PYTHON_VERSION=3.10
```

### Step 3: Set Up GitHub Secrets

1. **Get Azure Credentials**:

```powershell
az ad sp create-for-rbac \
  --name "equation-solver-github" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/equation-solver-rg \
  --sdk-auth
```

Copy the JSON output.

2. **Get ACR Credentials**:

```powershell
$ACR_USERNAME = az acr credential show --name equationsolver --query username -o tsv
$ACR_PASSWORD = az acr credential show --name equationsolver --query "passwords[0].value" -o tsv
Write-Host "Username: $ACR_USERNAME"
Write-Host "Password: $ACR_PASSWORD"
```

3. **Add Secrets to GitHub**:
   - Go to: https://github.com/sakshamsaxena22/Hand_Written_equation_solver/settings/secrets/actions
   - Click "New repository secret"
   - Add these secrets:
     - `AZURE_CREDENTIALS`: (paste JSON from step 1)
     - `ACR_USERNAME`: (from step 2)
     - `ACR_PASSWORD`: (from step 2)

### Step 4: Trigger Deployment

```powershell
# Commit and push the workflow
git add .github/workflows/azure-deploy.yml
git commit -m "Add Azure GitHub Actions deployment"
git push origin main
```

GitHub Actions will automatically:
1. Build Docker image
2. Push to Azure Container Registry
3. Deploy to App Service
4. Test health check

**Monitor**: https://github.com/sakshamsaxena22/Hand_Written_equation_solver/actions

---

## 🚀 Method 3: Azure CLI with GitHub Integration

Quickest method using CLI to set up GitHub integration:

```powershell
# Login
az login

# Create resources
$RG = "equation-solver-rg"
$APP = "equation-solver-app"

az group create --name $RG --location eastus

az acr create \
  --resource-group $RG \
  --name equationsolver \
  --sku Basic \
  --admin-enabled true

az appservice plan create \
  --name equation-solver-plan \
  --resource-group $RG \
  --is-linux \
  --sku B1

# Create Web App with GitHub deployment
az webapp create \
  --resource-group $RG \
  --plan equation-solver-plan \
  --name $APP \
  --deployment-source-url https://github.com/sakshamsaxena22/Hand_Written_equation_solver \
  --deployment-source-branch main \
  --docker-registry-server-url https://equationsolver.azurecr.io

# Enable continuous deployment
az webapp deployment source config \
  --name $APP \
  --resource-group $RG \
  --repo-url https://github.com/sakshamsaxena22/Hand_Written_equation_solver \
  --branch main \
  --manual-integration

# Configure container
$ACR_USER = az acr credential show --name equationsolver --query username -o tsv
$ACR_PASS = az acr credential show --name equationsolver --query "passwords[0].value" -o tsv

az webapp config container set \
  --name $APP \
  --resource-group $RG \
  --docker-custom-image-name equationsolver.azurecr.io/equation-solver:latest \
  --docker-registry-server-url https://equationsolver.azurecr.io \
  --docker-registry-server-user $ACR_USER \
  --docker-registry-server-password $ACR_PASS

# Configure app settings
az webapp config appsettings set \
  --name $APP \
  --resource-group $RG \
  --settings WEBSITES_PORT=8000 PYTHON_VERSION=3.10

# Trigger initial deployment
az acr build \
  --registry equationsolver \
  --image equation-solver:latest \
  --file Dockerfile \
  .

az webapp restart --name $APP --resource-group $RG
```

---

## 🔄 How Auto-Deployment Works

Once set up, every time you push to GitHub:

```
You push to GitHub (main branch)
        ↓
GitHub Actions triggered
        ↓
Azure builds Docker image
        ↓
Image pushed to Container Registry
        ↓
App Service pulls new image
        ↓
App automatically restarts
        ↓
New version live! ✅
```

**Deployment time**: 10-15 minutes per push

---

## 🧪 Testing Your Setup

### Verify Auto-Deployment

1. Make a small change to `azure_app.py`:
```python
# Change version number
version="2.0.1"  # was 2.0.0
```

2. Commit and push:
```powershell
git add azure_app.py
git commit -m "Test auto-deployment"
git push origin main
```

3. Watch GitHub Actions:
   - Go to: https://github.com/sakshamsaxena22/Hand_Written_equation_solver/actions
   - Click on your workflow run
   - Watch it build and deploy

4. After 10-15 minutes, test:
```powershell
curl https://equation-solver-app.azurewebsites.net/
# Should show version 2.0.1
```

---

## 📊 Monitoring Deployments

### GitHub Actions

View all deployments:
```
https://github.com/sakshamsaxena22/Hand_Written_equation_solver/actions
```

### Azure Portal

1. Go to your Web App
2. Click "Deployment Center"
3. See deployment history

### Azure CLI

```powershell
# View deployment logs
az webapp log tail --name equation-solver-app --resource-group equation-solver-rg

# View deployment history
az webapp deployment list-publishing-profiles \
  --name equation-solver-app \
  --resource-group equation-solver-rg
```

---

## 🔧 Common Issues

### Deployment Fails

**Check GitHub Actions logs:**
```
https://github.com/sakshamsaxena22/Hand_Written_equation_solver/actions
```

**Common fixes:**
1. Verify GitHub secrets are correct
2. Check Azure resource names match
3. Ensure ACR has admin access enabled

### App Won't Start

```powershell
# Check logs
az webapp log tail --name equation-solver-app --resource-group equation-solver-rg

# Common issues:
# - Port mismatch: ensure WEBSITES_PORT=8000
# - Memory limit: upgrade to B1 or higher
# - Startup timeout: increase in Portal
```

### Image Build Fails

**Check Dockerfile** is in repository root
**Verify** all dependencies in requirements.txt
**Increase** build timeout if needed

---

## 💰 Cost Breakdown

### GitHub Actions (Free Tier)
- 2,000 minutes/month free for public repos
- Your build uses ~15 minutes per deployment
- = ~133 deployments/month free

### Azure Resources
- Container Registry: $5/month
- App Service B1: $13/month
- **Total: ~$18/month**

---

## 🎯 Recommended Workflow

**For Development:**
```
1. Make changes locally
2. Test locally (optional)
3. Push to GitHub
4. GitHub Actions builds & deploys
5. Test on Azure
```

**For Production:**
```
1. Work in feature branch
2. Open Pull Request
3. Review changes
4. Merge to main
5. Auto-deploys to Azure
```

---

## 🔒 Security Best Practices

1. **Use GitHub Secrets** for all credentials
2. **Enable branch protection** on main branch
3. **Review deployments** before merging
4. **Use separate environments** (dev/staging/prod)
5. **Rotate credentials** regularly

---

## 📚 Additional Resources

- [GitHub Actions for Azure](https://github.com/Azure/actions)
- [Azure App Service GitHub Deployment](https://docs.microsoft.com/azure/app-service/deploy-github-actions)
- [Azure Container Registry](https://docs.microsoft.com/azure/container-registry/)

---

## ✅ Success Checklist

- [ ] Azure resources created
- [ ] GitHub connected to Azure
- [ ] Workflow file in `.github/workflows/`
- [ ] GitHub secrets configured
- [ ] First deployment successful
- [ ] Health check passes
- [ ] Demo page works
- [ ] Auto-deployment tested

---

## 🎉 You're Done!

Your GitHub repo is now connected to Azure with automatic deployments!

**Every push to main branch will:**
1. Build Docker image with PyTorch
2. Deploy to Azure automatically
3. Update your live application

**Access your app:**
- Demo: `https://equation-solver-app.azurewebsites.net/demo`
- API: `https://equation-solver-app.azurewebsites.net/docs`

**Monitor deployments:**
- GitHub: `https://github.com/sakshamsaxena22/Hand_Written_equation_solver/actions`
- Azure: `https://portal.azure.com`

---

**Happy deploying! 🚀**
