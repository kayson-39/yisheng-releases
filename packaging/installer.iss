; Compile via installer_build.py: an explicit, checked file list, never a tree glob.
#ifndef BuildOut
  #error Use installer_build.py with a verified stage and a new output directory
#endif
#ifndef CheckedFiles
  #error CheckedFiles is required
#endif
#ifndef ProductVersion
  #error ProductVersion is required
#endif
#define ModelName "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"
#define ModelHash "8f515bf4769f279a7fcf43e57446455a9d4de7f65b1bc9eddee76717e1ff7919"

[Setup]
; Verified against the deployed legacy Inno uninstall data (including the _is1 key).
AppId={{977B6A85-A0B2-4F52-8915-A388F2A7F289}
AppName=译声实时翻译
AppVersion={#ProductVersion}
VersionInfoVersion={#ProductVersion}.0
DefaultDirName={localappdata}\Programs\YishengTranslatorWhite
DefaultGroupName=译声实时翻译
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupArchitecture=x64
MinVersion=10.0
UsePreviousAppDir=not ValidationInstall
Uninstallable=not ValidationInstall
CreateUninstallRegKey=not ValidationInstall
DisableDirPage=no
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\译声实时翻译.exe
AppMutex={code:InstallationMutex}
CloseApplications=yes
RestartApplications=no
OutputDir={#BuildOut}
OutputBaseFilename=译声实时翻译安装器-{#ProductVersion}
SetupIconFile={#StageRoot}\app\jpsub\assets\icon.ico
InfoBeforeFile={#StageRoot}\使用说明.md
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=4
; Public format: a single double-clickable EXE, no companion volumes.
DiskSpanning=no
WizardStyle=modern

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: checkedonce; Check: not ValidationInstall
Name: "modeldownload"; Description: "缺失时下载 7B 翻译模型（约 4.25GB；已有模型会校验并复用）"; Flags: checkedonce; Check: not ValidationInstall

[Files]
#include CheckedFiles
Source: "{code:ReusableModelSource}"; DestDir: "{app}\app\models\llm"; DestName: "{#ModelName}"; ExternalSize: 4250296192; Hash: "{#ModelHash}"; Flags: external onlyifdoesntexist uninsneveruninstall; Check: CopyVerifiedModel
Source: "https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7/resolve/759d76b1745f4428308de6564c5ab710d4358b2e/{#ModelName}"; DestDir: "{app}\app\models\llm"; DestName: "{#ModelName}"; ExternalSize: 4250296192; Hash: "{#ModelHash}"; Flags: external download ignoreversion onlyifdoesntexist uninsneveruninstall; Check: DownloadMissingModel

[Icons]
Name: "{group}\译声实时翻译"; Filename: "{app}\译声实时翻译.exe"; WorkingDir: "{app}"; Check: not ValidationInstall
Name: "{userdesktop}\译声实时翻译"; Filename: "{app}\译声实时翻译.exe"; WorkingDir: "{app}"; Tasks: desktopicon; Check: not ValidationInstall

[Run]
Filename: "{app}\译声实时翻译.exe"; Description: "启动译声实时翻译"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent; Check: not ValidationInstall

[Code]
var
  CachedTarget, CachedModel: String;

function ValidationInstall: Boolean;
begin
  Result := ExpandConstant('{param:VALIDATE|0}') = '1';
end;

function InstallationMutex(Param: String): String;
begin
  if ValidationInstall then Result := ''
  else Result := 'Local\yisheng_white_release_single_instance';
end;

function TargetModel: String;
begin
  Result := ExpandConstant('{app}\app\models\llm\{#ModelName}');
end;

function VerifiedModel(Path: String): Boolean;
begin
  Result := False;
  if not FileExists(Path) then Exit;
  try
    Result := CompareText(GetSHA256OfFile(Path), '{#ModelHash}') = 0;
  except
    Log('Cannot verify existing model: ' + Path);
  end;
end;

function FindReusableModel: String;
var
  CurrentTarget, Candidate: String;
begin
  CurrentTarget := TargetModel;
  if CompareText(CachedTarget, CurrentTarget) = 0 then begin Result := CachedModel; Exit; end;
  CachedTarget := CurrentTarget;
  CachedModel := '';
  if VerifiedModel(CurrentTarget) then CachedModel := CurrentTarget;
  if CachedModel = '' then begin
    Candidate := ExpandConstant('{src}\{#ModelName}');
    if VerifiedModel(Candidate) then CachedModel := Candidate;
  end;
  if CachedModel = '' then begin
    Candidate := ExpandConstant('{localappdata}\Programs\YishengTranslatorWhite\app\models\llm\{#ModelName}');
    if (CompareText(Candidate, CurrentTarget) <> 0) and VerifiedModel(Candidate) then CachedModel := Candidate;
  end;
  Result := CachedModel;
end;

function ReusableModelSource(Param: String): String;
begin
  Result := '';
  if not ValidationInstall then Result := FindReusableModel;
  if Result = '' then Result := ExpandConstant('{tmp}\no-reusable-model.gguf');
end;

function CopyVerifiedModel: Boolean;
var
  Existing: String;
begin
  Result := False;
  if ValidationInstall or not WizardIsTaskSelected('modeldownload') then Exit;
  Existing := FindReusableModel;
  Result := (Existing <> '') and (CompareText(Existing, TargetModel) <> 0) and not FileExists(TargetModel);
end;

function DownloadMissingModel: Boolean;
begin
  Result := False;
  if ValidationInstall or not WizardIsTaskSelected('modeldownload') then Exit;
  Result := (FindReusableModel = '') and not FileExists(TargetModel);
end;

function ReadInstalledVersion(const FileName: String; var V: Int64): Boolean;
var
  Lines: TArrayOfString;
  I, P: Integer;
  L, Value: String;
begin
  Result := False;
  if not LoadStringsFromFile(FileName, Lines) then exit;
  for I := 0 to GetArrayLength(Lines) - 1 do begin
    L := Trim(Lines[I]);
    P := Pos('=', L);
    if (P > 0) and (Trim(Copy(L, 1, P-1)) = '__version__') then begin
      Value := Trim(Copy(L, P+1, Length(L)));
      if Length(Value) < 3 then exit;
      if (Value[1] <> '"') or (Value[Length(Value)] <> '"') then exit;
      Result := StrToVersion(Copy(Value, 2, Length(Value)-2), V);
      exit;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Installed, Candidate: Int64;
  AppPath, InitFile, LivePath: String;
begin
  Result := '';
  AppPath := ExpandConstant('{app}');
  if ValidationInstall then begin
    LivePath := AddBackslash(ExpandConstant('{localappdata}\Programs\YishengTranslatorWhite'));
    if (Pos(Lowercase(LivePath), Lowercase(AddBackslash(AppPath))) = 1) or
       (Pos(Lowercase(AddBackslash(AppPath)), Lowercase(LivePath)) = 1) or DirExists(AppPath) then begin
      Result := '隔离验证须通过 /DIR 指向不存在的新目录，不可选择现有安装或其他已有目录。';
    end;
    exit;
  end;
  if not DirExists(AppPath) then exit;
  InitFile := AppPath + '\app\jpsub\__init__.py';
  if not (FileExists(AppPath + '\译声实时翻译.exe') and
          FileExists(AppPath + '\unins000.dat') and
          ReadInstalledVersion(InitFile, Installed)) then begin
    Result := '安装目录已存在，但无法确认它是可升级的译声安装。为保护原文件，未覆盖。';
    exit;
  end;
  if not StrToVersion('{#ProductVersion}', Candidate) then begin
    Result := '安装包版本无效。';
    exit;
  end;
  if ComparePackedVersion(Installed, Candidate) > 0 then
    Result := '已安装更新的译声版本，不能使用此安装包降级。';
end;
