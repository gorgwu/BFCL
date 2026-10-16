param()

$ErrorActionPreference = "Stop"

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw "uv is required. Install it with: winget install --id=astral-sh.uv -e"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python was not found. Activate the BFCL virtual environment first."
}

$pythonPath = $pythonCommand.Source
$pythonVersion = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Could not query the active Python interpreter."
}
if ($pythonVersion -notmatch '^(3\.11|3\.12|3\.13)$') {
    throw "Use Python 3.11, 3.12, or 3.13 for this benchmark environment; found $pythonVersion."
}

$virtualEnv = $env:VIRTUAL_ENV
if (-not $virtualEnv) {
    throw "Activate the BFCL virtual environment before running this installer."
}

$bfclRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $bfclRoot)
$swarmRoot = if ($env:BFCL_JIUWENSWARM_ROOT) {
    $env:BFCL_JIUWENSWARM_ROOT
} else {
    Join-Path $workspaceRoot "jiuwenswarm"
}
$coreRoot = if ($env:BFCL_AGENT_CORE_ROOT) {
    $env:BFCL_AGENT_CORE_ROOT
} else {
    Join-Path $workspaceRoot "agent-core"
}

foreach ($path in @($swarmRoot, $coreRoot)) {
    if (-not (Test-Path -LiteralPath (Join-Path $path "pyproject.toml"))) {
        throw "Expected repository pyproject.toml was not found under $path. Set BFCL_JIUWENSWARM_ROOT / BFCL_AGENT_CORE_ROOT for non-sibling checkouts."
    }
}

Write-Host "Syncing JiuwenSwarm's locked runtime dependencies into $virtualEnv"
& $uvCommand.Source sync --project $swarmRoot --active --no-default-groups --no-install-project --no-install-package openjiuwen
if ($LASTEXITCODE -ne 0) {
    throw "JiuwenSwarm dependency sync failed."
}

Write-Host "Installing the checked-out OpenJiuwen source and its declared dependencies"
& $uvCommand.Source pip install --python $pythonPath --editable $coreRoot
if ($LASTEXITCODE -ne 0) {
    throw "Local agent-core installation failed."
}

Write-Host "Installing BFCL and its compatible parser / CLI dependencies"
& $uvCommand.Source pip install --python $pythonPath --editable $bfclRoot
if ($LASTEXITCODE -ne 0) {
    throw "BFCL installation failed."
}

Write-Host "Dependency installation completed. BFCL's adapter loads the checked-out JiuwenSwarm source from its sibling directory."
