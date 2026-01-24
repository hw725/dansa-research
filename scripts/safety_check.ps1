#!/usr/bin/env pwsh
<#
.SYNOPSIS
    CSP 프로젝트 패키지 보안 취약점 점검 스크립트

.DESCRIPTION
    pip-audit를 사용하여 requirements.txt의 보안 취약점을 확인합니다.
    (safety는 유료화되어 pip-audit 사용 권장)

.EXAMPLE
    ./scripts/safety_check.ps1
    ./scripts/safety_check.ps1 -Docker
#>

param(
    [switch]$Docker,      # Docker 컨테이너 내에서 실행
    [switch]$Fix,         # 취약점 자동 수정 시도
    [switch]$Verbose      # 상세 출력
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " CSP 패키지 보안 점검" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($Docker) {
    Write-Host "[Docker] 컨테이너 내에서 점검 실행..." -ForegroundColor Yellow
    
    # Docker 컨테이너에서 pip-audit 설치 및 실행
    $auditCmd = "pip install --quiet pip-audit && pip-audit"
    if ($Fix) {
        $auditCmd += " --fix"
    }
    if ($Verbose) {
        $auditCmd += " --desc"
    }
    
    docker compose run --rm csp bash -c $auditCmd
    $exitCode = $LASTEXITCODE
} else {
    Write-Host "[로컬] .venv 환경에서 점검 실행..." -ForegroundColor Yellow
    
    # 가상환경 활성화 확인
    $venvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $venvPath) {
        . $venvPath
    } else {
        Write-Warning ".venv가 없습니다. 전역 Python 사용."
    }
    
    # pip-audit 설치 확인
    $pipAudit = pip show pip-audit 2>$null
    if (-not $pipAudit) {
        Write-Host "pip-audit 설치 중..." -ForegroundColor Yellow
        pip install pip-audit --quiet
    }
    
    # 점검 실행
    $requirementsPath = Join-Path $ProjectRoot "requirements.txt"
    $auditArgs = @("-r", $requirementsPath)
    
    if ($Fix) {
        $auditArgs += "--fix"
    }
    if ($Verbose) {
        $auditArgs += "--desc"
    }
    
    pip-audit @auditArgs
    $exitCode = $LASTEXITCODE
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "✅ 취약점이 발견되지 않았습니다." -ForegroundColor Green
} else {
    Write-Host "⚠️  취약점이 발견되었습니다. 위 내용을 확인하세요." -ForegroundColor Red
    Write-Host "   자동 수정을 시도하려면: ./scripts/safety_check.ps1 -Fix" -ForegroundColor Yellow
}

exit $exitCode
