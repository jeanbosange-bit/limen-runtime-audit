$ErrorActionPreference = "Stop"

$downloads = Join-Path $HOME "Downloads"
$files = @(
    "extract_limen_trajectory.py",
    "test_extract_limen_trajectory.py",
    "03_run_limen_extractor_on_jetson.sh"
)

foreach ($name in $files) {
    $path = Join-Path $downloads $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Fichier introuvable : $path"
    }
    Start-Process notepad.exe -ArgumentList "`"$path`""
}
