$ErrorActionPreference = "Stop"

$Jetson = "jeanbatuli@ubuntu"
$RemoteDirectory = "/home/jeanbatuli/limen-runtime-audit/scripts"
$LocalScript = Join-Path $HOME "Downloads\02_run_limen_replication_on_jetson.sh"

if (-not (Test-Path -LiteralPath $LocalScript)) {
    throw "Fichier absent : $LocalScript"
}

Write-Host "Empreinte SHA-256 locale :"
Get-FileHash -Algorithm SHA256 -LiteralPath $LocalScript

Write-Host "Création ou vérification du dossier distant : $RemoteDirectory"
ssh $Jetson "mkdir -p '$RemoteDirectory'"

Write-Host "Transfert du lanceur de réplication"
scp $LocalScript "${Jetson}:${RemoteDirectory}/02_run_limen_replication_on_jetson.sh"

Write-Host "Vérification distante"
ssh $Jetson "chmod +x '$RemoteDirectory/02_run_limen_replication_on_jetson.sh' && sha256sum '$RemoteDirectory/02_run_limen_replication_on_jetson.sh'"

Write-Host ""
Write-Host "Transfert terminé. Sur le Jetson, lance :"
Write-Host "bash ~/limen-runtime-audit/scripts/02_run_limen_replication_on_jetson.sh"
