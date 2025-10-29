# ============================================
# Azure Deployment Script for Equation Solver
# ============================================
# This script deploys the full ML application to Azure App Service

Write-Host "🚀 Azure Deployment Script for Hand-Written Equation Solver" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Configuration
$RESOURCE_GROUP = "equation-solver-rg"
$LOCATION = "eastus"
$ACR_NAME = "equationsolver$(Get-Random -Maximum 9999)"  # Add random suffix for uniqueness
$APP_PLAN = "equation-solver-plan"
$APP_NAME = "equation-solver-app-$(Get-Random -Maximum 9999)"  # Add random suffix for uniqueness
$IMAGE_NAME = "equation-solver"

Write-Host "`n📋 Configuration:" -ForegroundColor Yellow
Write-Host "  Resource Group: $RESOURCE_GROUP" -ForegroundColor White
Write-Host "  Location: $LOCATION" -ForegroundColor White
Write-Host "  ACR Name: $ACR_NAME" -ForegroundColor White
Write-Host "  App Name: $APP_NAME" -ForegroundColor White
Write-Host ""

# Confirm before proceeding
$confirm = Read-Host "Do you want to proceed with deployment? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Deployment cancelled." -ForegroundColor Red
    exit
}

# Step 1: Login to Azure
Write-Host "`n[1/9] Logging in to Azure..." -ForegroundColor Green
az login

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Azure login failed" -ForegroundColor Red
    exit 1
}

# Step 2: Create Resource Group
Write-Host "`n[2/9] Creating resource group..." -ForegroundColor Green
az group create --name $RESOURCE_GROUP --location $LOCATION

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create resource group" -ForegroundColor Red
    exit 1
}

# Step 3: Create Container Registry
Write-Host "`n[3/9] Creating Azure Container Registry (this may take 2-3 minutes)..." -ForegroundColor Green
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Container Registry" -ForegroundColor Red
    exit 1
}

# Step 4: Build and push Docker image
Write-Host "`n[4/9] Building Docker image (this will take 10-15 minutes)..." -ForegroundColor Green
Write-Host "☕ Time for a coffee break!" -ForegroundColor Yellow

az acr build `
    --registry $ACR_NAME `
    --image "$IMAGE_NAME:latest" `
    --file Dockerfile `
    .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build Docker image" -ForegroundColor Red
    exit 1
}

# Step 5: Create App Service Plan
Write-Host "`n[5/9] Creating App Service Plan..." -ForegroundColor Green
az appservice plan create `
    --name $APP_PLAN `
    --resource-group $RESOURCE_GROUP `
    --is-linux `
    --sku B1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create App Service Plan" -ForegroundColor Red
    exit 1
}

# Step 6: Create Web App
Write-Host "`n[6/9] Creating Web App..." -ForegroundColor Green
az webapp create `
    --resource-group $RESOURCE_GROUP `
    --plan $APP_PLAN `
    --name $APP_NAME `
    --deployment-container-image-name "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Web App" -ForegroundColor Red
    exit 1
}

# Step 7: Configure Web App
Write-Host "`n[7/9] Configuring Web App..." -ForegroundColor Green

# Get ACR credentials
Write-Host "  Getting ACR credentials..." -ForegroundColor White
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

# Configure container settings
Write-Host "  Configuring container..." -ForegroundColor White
az webapp config container set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-custom-image-name "$ACR_NAME.azurecr.io/$IMAGE_NAME:latest" `
    --docker-registry-server-url "https://$ACR_NAME.azurecr.io" `
    --docker-registry-server-user $ACR_USERNAME `
    --docker-registry-server-password $ACR_PASSWORD

# Configure app settings
Write-Host "  Configuring app settings..." -ForegroundColor White
az webapp config appsettings set `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --settings WEBSITES_PORT=8000 PYTHON_VERSION=3.10

# Enable logging
Write-Host "  Enabling logging..." -ForegroundColor White
az webapp log config `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --docker-container-logging filesystem

# Enable Always On (requires B1 or higher)
Write-Host "  Enabling Always On..." -ForegroundColor White
az webapp config set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --always-on true

# Step 8: Restart app
Write-Host "`n[8/9] Restarting app..." -ForegroundColor Green
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

# Step 9: Get URL and test
Write-Host "`n[9/9] Finalizing deployment..." -ForegroundColor Green
$APP_URL = az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName -o tsv

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host "`n📍 Your application is deployed at:" -ForegroundColor Yellow
Write-Host "  App URL:  https://$APP_URL" -ForegroundColor Cyan
Write-Host "  Demo:     https://$APP_URL/demo" -ForegroundColor Cyan
Write-Host "  API Docs: https://$APP_URL/docs" -ForegroundColor Cyan
Write-Host "  Health:   https://$APP_URL/health" -ForegroundColor Cyan

Write-Host "`n⏳ Waiting for app to start (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host "`n🧪 Testing health check..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "https://$APP_URL/health" -Method Get
    Write-Host "✅ Health check passed!" -ForegroundColor Green
    Write-Host "   Status: $($response.status)" -ForegroundColor White
    Write-Host "   Solver: $($response.solver)" -ForegroundColor White
    Write-Host "   ML Models: $($response.ml_models)" -ForegroundColor White
} catch {
    Write-Host "⚠️  Health check failed (app may still be starting)" -ForegroundColor Yellow
    Write-Host "   Please wait a few more minutes and try: https://$APP_URL/health" -ForegroundColor White
}

Write-Host "`n📝 Resource Details:" -ForegroundColor Yellow
Write-Host "  Resource Group: $RESOURCE_GROUP" -ForegroundColor White
Write-Host "  Container Registry: $ACR_NAME" -ForegroundColor White
Write-Host "  App Service Plan: $APP_PLAN (B1 Basic)" -ForegroundColor White
Write-Host "  Web App: $APP_NAME" -ForegroundColor White

Write-Host "`n💡 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Visit https://$APP_URL/demo to test the application" -ForegroundColor White
Write-Host "  2. Check logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP" -ForegroundColor White
Write-Host "  3. View in Azure Portal: https://portal.azure.com" -ForegroundColor White

Write-Host "`n💰 Estimated Cost: ~`$18/month (Basic tier)" -ForegroundColor Yellow
Write-Host "   - Container Registry Basic: `$5/month" -ForegroundColor White
Write-Host "   - App Service B1: `$13/month" -ForegroundColor White

Write-Host "`n🗑️  To delete all resources:" -ForegroundColor Red
Write-Host "   az group delete --name $RESOURCE_GROUP --yes --no-wait" -ForegroundColor White

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "Happy solving! 🧮" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
