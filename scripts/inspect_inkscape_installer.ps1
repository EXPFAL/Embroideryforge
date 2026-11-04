$ErrorActionPreference = 'Continue'
$path = 'C:\Users\Administrator\Downloads\inkscape.exe'
Write-Host "exists:" (Test-Path -LiteralPath $path)
if (-not (Test-Path -LiteralPath $path)) { exit 1 }

$item = Get-Item -LiteralPath $path
Write-Host "full:" $item.FullName
Write-Host "size_bytes:" $item.Length
Write-Host "size_mb:" ([math]::Round($item.Length / 1MB, 2))
Write-Host "created:" $item.CreationTime
Write-Host "modified:" $item.LastWriteTime

try {
  $sig = Get-AuthenticodeSignature -LiteralPath $path
  Write-Host "signature_status:" $sig.Status
  Write-Host "signer:" $sig.SignerCertificate.Subject
} catch {
  Write-Host "signature_error:" $_
}

# PE product version strings if available
try {
  $vi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)
  Write-Host "file_description:" $vi.FileDescription
  Write-Host "product_name:" $vi.ProductName
  Write-Host "company:" $vi.CompanyName
  Write-Host "file_version:" $vi.FileVersion
  Write-Host "original_filename:" $vi.OriginalFilename
} catch {
  Write-Host "version_info_error:" $_
}

# Check winget availability
Write-Host "winget:" (Get-Command winget -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
Write-Host "======= DONE ======="
