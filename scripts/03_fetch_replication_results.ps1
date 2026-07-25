$ErrorActionPreference = "Stop"

$Jetson = "jeanbatuli@ubuntu"
$RemoteDirectory = "/home/jeanbatuli/limen-runtime-audit/scripts/limen_extraction_output/tinyllama_replication_20260725"
$LocalDirectory = Join-Path $HOME "Downloads\tinyllama_replication_20260725"

New-Item -ItemType Directory -Force -Path $LocalDirectory | Out-Null

foreach ($Name in @("trajectory.npz", "trajectory.metadata.json", "comparison.json", "extraction.log")) {
    Write-Host "Téléchargement : $Name"
    scp "${Jetson}:${RemoteDirectory}/${Name}" (Join-Path $LocalDirectory $Name)
}

Write-Host ""
Write-Host "Fichiers récupérés dans : $LocalDirectory"
Get-ChildItem -LiteralPath $LocalDirectory |
    Select-Object Name, Length, LastWriteTime
explorer $LocalDirectory
