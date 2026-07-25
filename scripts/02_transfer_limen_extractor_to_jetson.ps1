$ErrorActionPreference = "Stop"

$downloads = Join-Path $HOME "Downloads"
$remote = "jeanbatuli@ubuntu"
$remoteDirectory = "/home/jeanbatuli/limen-runtime-audit/scripts"
$files = @(
    "extract_limen_trajectory.py",
    "test_extract_limen_trajectory.py",
    "03_run_limen_extractor_on_jetson.sh"
)

Write-Host "=== EMPREINTES SHA-256 LOCALES ==="
foreach ($name in $files) {
    $path = Join-Path $downloads $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Fichier introuvable : $path"
    }
    Get-FileHash -Algorithm SHA256 -LiteralPath $path
}

Write-Host "=== CREATION DU DOSSIER DISTANT ==="
ssh $remote "mkdir -p '$remoteDirectory'"
if ($LASTEXITCODE -ne 0) {
    throw "Impossible de créer le dossier distant."
}

foreach ($name in $files) {
    $path = Join-Path $downloads $name
    Write-Host "Transfert : $path"
    scp -- $path "${remote}:${remoteDirectory}/${name}"
    if ($LASTEXITCODE -ne 0) {
        throw "Échec SCP pour $name"
    }
}

Write-Host "=== VERIFICATION DISTANTE ==="
ssh $remote "cd '$remoteDirectory' && chmod +x 03_run_limen_extractor_on_jetson.sh && sha256sum extract_limen_trajectory.py test_extract_limen_trajectory.py 03_run_limen_extractor_on_jetson.sh"
if ($LASTEXITCODE -ne 0) {
    throw "La vérification distante a échoué."
}

Write-Host ""
Write-Host "Transfert terminé."
Write-Host "Sur le Jetson, lance :"
Write-Host "bash ~/limen-runtime-audit/scripts/03_run_limen_extractor_on_jetson.sh"
