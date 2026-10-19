; AstroFlow's Windows installer. Built by app/windows/build_installer.ps1, which
; defines Version, Stage (the install layout), Out and Src (the repo).
#ifndef Version
  #define Version "0.0"
#endif

[Setup]
AppId={{7E1C0C62-5B1E-4E7D-9B1A-4B7C2E3F8A10}
AppName=AstroFlow
AppVersion={#Version}
AppPublisher=Stylistic7
DefaultDirName={autopf}\AstroFlow
DefaultGroupName=AstroFlow
DisableProgramGroupPage=yes
OutputDir={#Out}
OutputBaseFilename=AstroFlow-Setup-{#Version}
SetupIconFile={#Src}\app\windows\icon.ico
UninstallDisplayIcon={app}\icon.ico
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Several gigabytes, most of it index files that barely compress: fast, not solid.
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "asiair"; Description: "Let Windows read an ASIAIR's shares (guest SMB access on, SMB signing not required) so AstroFlow can download frames by itself"; Flags: unchecked

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\AstroFlow"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\AstroFlow\app\windows\AstroFlow.pyw"""; WorkingDir: "{app}\AstroFlow\core"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\AstroFlow"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\AstroFlow\app\windows\AstroFlow.pyw"""; WorkingDir: "{app}\AstroFlow\core"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
; The programs that listen on ports (the web servers, Ray), allowed through the firewall so Windows never asks.
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""AstroFlow"" dir=in action=allow program=""{app}\python\python.exe"""; Flags: runhidden
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""AstroFlow Ray"" dir=in action=allow program=""{app}\python\Lib\site-packages\ray\core\src\ray\raylet\raylet.exe"""; Flags: runhidden
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""AstroFlow Ray GCS"" dir=in action=allow program=""{app}\python\Lib\site-packages\ray\core\src\ray\gcs\gcs_server.exe"""; Flags: runhidden
Filename: "reg"; Parameters: "add HKLM\SYSTEM\CurrentControlSet\Services\LanmanWorkstation\Parameters /v AllowInsecureGuestAuth /t REG_DWORD /d 1 /f"; Tasks: asiair; Flags: runhidden
Filename: "powershell"; Parameters: "-NoProfile -Command ""Set-SmbClientConfiguration -RequireSecuritySignature $false -Force"""; Tasks: asiair; Flags: runhidden
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\AstroFlow\app\windows\AstroFlow.pyw"""; WorkingDir: "{app}\AstroFlow\core"; Description: "Start AstroFlow"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""AstroFlow"""; Flags: runhidden; RunOnceId: "fw1"
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""AstroFlow Ray"""; Flags: runhidden; RunOnceId: "fw2"
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""AstroFlow Ray GCS"""; Flags: runhidden; RunOnceId: "fw3"
