$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================="
Write-Host " FRAUD DETECTION - DOCKER TEST"
Write-Host "========================================="
Write-Host ""

Write-Host "[1/5] Derrubando containers..."
docker compose down

Write-Host ""
Write-Host "[2/5] Construindo imagem..."
docker compose build

Write-Host ""
Write-Host "[3/5] Iniciando containers..."
docker compose up -d

Write-Host ""
Write-Host "[4/5] Verificando status..."
docker compose ps

Write-Host ""
Write-Host "[5/5] Testando API..."

$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {

    try {
        $response = Invoke-WebRequest `
            -Uri "http://localhost:8000/health" `
            -Method GET `
            -TimeoutSec 2

        if ($response.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "API respondeu corretamente!" -ForegroundColor Green
            break
        }
    }
    catch {
        Write-Host "Aguardando API..."
    }

    $attempt++
    Start-Sleep -Seconds 1
}

if ($attempt -eq $maxAttempts) {

    Write-Host ""
    Write-Host "ERRO: API não respondeu." -ForegroundColor Red

    docker compose logs

    docker compose down

    exit 1
}

Write-Host ""
Write-Host "========================================="
Write-Host " DOCKER TEST PASSOU"
Write-Host "========================================="
Write-Host ""

exit 0