param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$installRoot = 'C:\Users\pz730\AppData\Local\Programs\YishengTranslatorWhite'

# Only touch the exact reviewed 0.10.9 files. A later/custom install must not
# be silently overwritten. Models, config, history and the launcher are not targets.
$expected = [ordered]@{
    'app\jpsub\__init__.py' = '36A4399EA30B42129B1C7A00E1819287F285C09561B81C2546690FDD0307CAB5'
    'app\jpsub\core\pipeline.py' = '29BC5B86EC73ED242830B7145D6013EE42F7AB3CF369A94E41A00776ECE836CA'
    'app\jpsub\core\history.py' = '86EB20A1A78DE788DD156DC6A29829D2405F36E9A2FEB17DC3F908620C651262'
    'app\jpsub\ui\window.py' = '07EC06AB4EB3BFFCBAB51D170B5E55B33B1955EBA3847246AF494E6C186391BE'
    'app\jpsub\ui\transcript.py' = 'DDE71611BBF7FB8885D7D25E211674F11C68C50FA4E0A16CFF9168C60B767712'
    'app\jpsub\ui\text_translate.py' = 'E56E456D19E18FF7F1A82C89F8F5D28E87D2C7BA881C3A8BE289DC7869C51EB9'
    'app\README.md' = '65D39724109CA2C63C8F105458DA8B3C8184CD0B3AB28526153E22A832E5C550'
    '使用说明.md' = '65D39724109CA2C63C8F105458DA8B3C8184CD0B3AB28526153E22A832E5C550'
}
function Get-PatchSource([string]$relative) {
    if ($relative -eq '使用说明.md') { return (Join-Path $workspace 'README.md') }
    return (Join-Path $workspace $relative.Substring(4))
}
function Assert-AppClosed {
    $jobs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"
    if ($jobs | Where-Object { $_.ExecutablePath -like "$installRoot\*" }) {
        throw '译声仍在运行，没有覆盖安装文件。请先正常退出译声。'
    }
}
$records = foreach ($relative in $expected.Keys) {
    $currentHash = (Get-FileHash -LiteralPath (Join-Path $installRoot $relative)).Hash
    $sourceHash = (Get-FileHash -LiteralPath (Get-PatchSource $relative)).Hash
    [pscustomobject]@{ relative=$relative; old=$expected[$relative]; new=$sourceHash; current=$currentHash }
}
if (-not ($records | Where-Object { $_.current -ne $_.new })) {
    Write-Output 'Installed files already match this patch.'
    return
}
foreach ($record in $records) {
    if ($record.current -ne $record.old) { throw "0.10.9 基线不匹配，未写入：$($record.relative)" }
}
if ($CheckOnly) { Write-Output "Ready: $($records.Count) files match 0.10.9."; return }
Assert-AppClosed
$backupRoot = Join-Path $workspace ('artifacts\backup-0.10.9-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $backupRoot | Out-Null
foreach ($record in $records) {
    $backupFile = Join-Path $backupRoot $record.relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $backupFile) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $installRoot $record.relative) -Destination $backupFile
    if ((Get-FileHash -LiteralPath $backupFile).Hash -ne $record.old) { throw "备份校验失败：$backupFile" }
}
$records | Select-Object relative,old,new | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $backupRoot 'manifest.json') -Encoding utf8
Assert-AppClosed
$touched = @()
try {
    foreach ($record in $records) {
        $target = Join-Path $installRoot $record.relative
        if ((Get-FileHash -LiteralPath $target).Hash -ne $record.old) { throw "目标在备份后变更：$target" }
        $touched += $record
        Copy-Item -LiteralPath (Get-PatchSource $record.relative) -Destination $target
        if ((Get-FileHash -LiteralPath $target).Hash -ne $record.new) { throw "部署校验失败：$target" }
    }
} catch {
    foreach ($record in $touched) {
        Copy-Item -LiteralPath (Join-Path $backupRoot $record.relative) -Destination (Join-Path $installRoot $record.relative)
    }
    throw
}
Write-Output "Installed and verified 0.10.10 ($($records.Count) files); backup: $backupRoot"
