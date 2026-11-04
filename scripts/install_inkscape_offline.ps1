$ErrorActionPreference = 'Stop'

$msiUrl = 'https://media.inkscape.org/dl/resources/file/inkscape-1.4.4_2026-05-05_dcaf3e7-x64.signed_xMx7DJV.msi'
$expectedSha = '2fa033e95113ae83d6430b2479e7e63e4266d0632a5b03f99ee3db9360e3b8b8'
$out = 'C:\Users\Administrator\Downloads\inkscape-1.4.4-x64.msi'

Write-Host "Downloading official MSI (not Store)..."
Write-Host "URL: $msiUrl"
Write-Host "OUT: $out"

curl.exe -L --fail --retry 3 -o $out $msiUrl
if ($LASTEXITCODE -ne 0) { throw "curl failed: $LASTEXITCODE" }

$actual = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "size_mb:" ([math]::Round((Get-Item -LiteralPath $out).Length / 1MB, 2))
Write-Host "sha256:" $actual
if ($actual -ne $expectedSha.ToLowerInvariant()) {
    throw "SHA256 mismatch. expected=$expectedSha actual=$actual"
}
Write-Host "SHA256 OK"

Write-Host "Installing via winget source=winget (avoids msstore)..."
winget install --id Inkscape.Inkscape -e --source winget --accept-package-agreements --accept-source-agreements
if ($LASTEXITCODE -ne 0) {
    Write-Host "winget install failed ($LASTEXITCODE). You can double-click the MSI instead:"
    Write-Host $out
    exit $LASTEXITCODE
}

Write-Host "Install finished."
Get-Command inkscape -ErrorAction SilentlyContinue | Format-List
$paths = @(
    'C:\Program Files\Inkscape\bin\inkscape.exe',
    'C:\Program Files\Inkscape\inkscape.exe'
)
foreach ($p in $paths) {
    if (Test-Path -LiteralPath $p) { Write-Host "found:" $p }
}
Write-Host "DONE"
