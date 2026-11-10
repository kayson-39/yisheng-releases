param([Parameter(Mandatory=$true)][string]$BackupDirectory)
$ErrorActionPreference = 'Stop'
$installRoot = 'C:\Users\pz730\AppData\Local\Programs\YishengTranslatorWhite'
$running = Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'pythonw.exe' -and $_.ExecutablePath -like "$installRoot\*"}
if ($running) { throw '请先退出译声软件。' }
$backup = (Resolve-Path -LiteralPath $BackupDirectory).Path
$records = Get-Content -LiteralPath (Join-Path $backup 'manifest.json') -Raw | ConvertFrom-Json
foreach ($record in $records) {
    $target = [IO.Path]::GetFullPath((Join-Path $installRoot $record.relative))
    $source = [IO.Path]::GetFullPath((Join-Path $backup $record.relative))
    if (-not $target.StartsWith($installRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $source.StartsWith($backup + '\', [StringComparison]::OrdinalIgnoreCase)) { throw '备份路径越界' }
    if ((Get-FileHash -LiteralPath $source).Hash -ne $record.old) { throw "备份校验失败：$source" }
    if ((Get-FileHash -LiteralPath $target).Hash -ne $record.new) { throw '安装文件已再次变更，停止回退以避免覆盖新修改。' }
}
foreach ($record in $records) {
    Copy-Item -LiteralPath (Join-Path $backup $record.relative) -Destination (Join-Path $installRoot $record.relative)
}
Write-Output '已恢复备份版本；个人配置、模型和历史未修改。'
