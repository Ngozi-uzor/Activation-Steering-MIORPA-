# Builds a self-contained Colab notebook: embeds the miorpa package as base64
# so the notebook is the only file that needs uploading. Dataset still comes
# from Hugging Face at runtime.
#
# Run this again after editing anything in miorpa/ to refresh the embedded copy.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $root 'miorpa'
$in   = Join-Path $root 'MIORPA_Colab_Source.ipynb'
$out  = Join-Path $root 'MIORPA_Colab_FullSet.ipynb'

# --- base64 each module -----------------------------------------------------
$lines = New-Object System.Collections.ArrayList
[void]$lines.Add("# Self-contained bootstrap. The miorpa package is embedded in this cell as")
[void]$lines.Add("# base64, so this notebook is the only file you need. The PRISM dataset is")
[void]$lines.Add("# downloaded from Hugging Face in the next step.")
[void]$lines.Add("#")
[void]$lines.Add("# If a miorpa/ folder is already present (Drive, or a previous run), that one")
[void]$lines.Add("# is used instead so local edits are not overwritten.")
[void]$lines.Add("")
[void]$lines.Add("import base64, os, sys, glob, importlib")
[void]$lines.Add("")
[void]$lines.Add("_EMBEDDED = {")

Get-ChildItem $src -Filter *.py | Sort-Object Name | ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $b64   = [System.Convert]::ToBase64String($bytes)
    [void]$lines.Add("    '$($_.Name)': '$b64',")
    Write-Output ("embedded {0,-18} {1,7:N0} bytes -> {2,7:N0} b64" -f $_.Name, $bytes.Length, $b64.Length)
}

[void]$lines.Add("}")
[void]$lines.Add("")
[void]$lines.Add("def _find_existing():")
[void]$lines.Add("    cands = ['/content/drive/MyDrive/MIORPA PROJECT',")
[void]$lines.Add("             '/content/MIORPA PROJECT', os.getcwd()]")
[void]$lines.Add("    cands += [os.path.dirname(os.path.dirname(p))")
[void]$lines.Add("              for p in glob.glob('/content/**/miorpa/config.py', recursive=True)]")
[void]$lines.Add("    for c in cands:")
[void]$lines.Add("        if c and os.path.isfile(os.path.join(c, 'miorpa', 'config.py')):")
[void]$lines.Add("            return c")
[void]$lines.Add("    return None")
[void]$lines.Add("")
[void]$lines.Add("# Mount Drive and use it as the working directory. Everything the pipeline")
[void]$lines.Add("# writes - dataset, vectors, generations, evaluation CSVs - then lands in")
[void]$lines.Add("# Drive as it's produced, so it survives a disconnect and syncs to your")
[void]$lines.Add("# computer through Drive itself instead of sitting on the throwaway runtime")
[void]$lines.Add("# disk waiting to be manually downloaded before the session ends.")
[void]$lines.Add("IN_COLAB = True")
[void]$lines.Add("try:")
[void]$lines.Add("    import google.colab  # noqa: F401")
[void]$lines.Add("    if not os.path.isdir('/content/drive/MyDrive'):")
[void]$lines.Add("        from google.colab import drive")
[void]$lines.Add("        drive.mount('/content/drive')")
[void]$lines.Add("except ImportError:")
[void]$lines.Add("    IN_COLAB = False")
[void]$lines.Add("")
[void]$lines.Add("# Drive wins whenever it is available, ahead of anything _find_existing")
[void]$lines.Add("# turns up. Its candidate list includes the working directory and a glob")
[void]$lines.Add("# under /content, so a stray miorpa folder on the temporary disk could")
[void]$lines.Add("# otherwise capture the run and send every result somewhere that does not")
[void]$lines.Add("# survive the session.")
[void]$lines.Add("if IN_COLAB and os.path.isdir('/content/drive/MyDrive'):")
[void]$lines.Add("    PROJECT_DIR = '/content/drive/MyDrive/MIORPA PROJECT'")
[void]$lines.Add("else:")
[void]$lines.Add("    PROJECT_DIR = _find_existing()")
[void]$lines.Add("    if PROJECT_DIR is None:")
[void]$lines.Add("        PROJECT_DIR = '/content' if os.path.isdir('/content') else os.getcwd()")
[void]$lines.Add("")
[void]$lines.Add("# The embedded copy always wins. An earlier version preferred whatever")
[void]$lines.Add("# package was already on disk, to protect local edits - but once the")
[void]$lines.Add("# project lives in Drive that folder survives between sessions, so every")
[void]$lines.Add("# run silently executed stale code while the notebook looked current.")
[void]$lines.Add("# That cost a full judging run. Now anything that differs is overwritten")
[void]$lines.Add("# and named, so a mismatch is visible instead of silent.")
[void]$lines.Add("target = os.path.join(PROJECT_DIR, 'miorpa')")
[void]$lines.Add("os.makedirs(target, exist_ok=True)")
[void]$lines.Add("written, refreshed = [], []")
[void]$lines.Add("for name, blob in _EMBEDDED.items():")
[void]$lines.Add("    payload = base64.b64decode(blob)")
[void]$lines.Add("    path = os.path.join(target, name)")
[void]$lines.Add("    if os.path.isfile(path):")
[void]$lines.Add("        with open(path, 'rb') as fh:")
[void]$lines.Add("            if fh.read() == payload:")
[void]$lines.Add("                continue")
[void]$lines.Add("        refreshed.append(name)")
[void]$lines.Add("    else:")
[void]$lines.Add("        written.append(name)")
[void]$lines.Add("    with open(path, 'wb') as fh:")
[void]$lines.Add("        fh.write(payload)")
[void]$lines.Add("")
[void]$lines.Add("# Three distinct outcomes, each said plainly. Collapsing 'wrote it fresh'")
[void]$lines.Add("# into 'already matches' would be the same class of misleading")
[void]$lines.Add("# confirmation that let stale code run unnoticed in the first place.")
[void]$lines.Add("if written:")
[void]$lines.Add("    print(f'wrote {len(written)} modules to {target}')")
[void]$lines.Add("if refreshed:")
[void]$lines.Add("    print('REFRESHED stale modules: ' + ', '.join(sorted(refreshed)))")
[void]$lines.Add("    print('the copy on disk was older than this notebook and has been replaced')")
[void]$lines.Add("if not written and not refreshed:")
[void]$lines.Add("    print(f'package in {target} already matches the notebook')")
[void]$lines.Add("")
[void]$lines.Add("os.chdir(PROJECT_DIR)")
[void]$lines.Add("if PROJECT_DIR not in sys.path:")
[void]$lines.Add("    sys.path.insert(0, PROJECT_DIR)")
[void]$lines.Add("")
[void]$lines.Add("# Drop any half-imported copy so a re-run picks up the files just written.")
[void]$lines.Add("for mod in [m for m in list(sys.modules) if m == 'miorpa' or m.startswith('miorpa.')]:")
[void]$lines.Add("    del sys.modules[mod]")
[void]$lines.Add("")
[void]$lines.Add("print('working in', PROJECT_DIR)")
[void]$lines.Add("print('package present:', os.path.isfile('miorpa/config.py'))")
[void]$lines.Add("if PROJECT_DIR.startswith('/content/drive/'):")
[void]$lines.Add("    print('results are saving to Google Drive - safe if this session drops')")
[void]$lines.Add("elif IN_COLAB:")
[void]$lines.Add("    print('WARNING: results are saving to the temporary Colab disk, not Drive.')")
[void]$lines.Add("    print('They will be lost if this session disconnects. Download them with')")
[void]$lines.Add("    print('the export cell before ending the session.')")
[void]$lines.Add("")
[void]$lines.Add("have = {f: os.path.isfile(f'dataset/{f}.jsonl')")
[void]$lines.Add("        for f in ('survey', 'conversations', 'utterances')}")
[void]$lines.Add("print('dataset present:', all(have.values()), have)")
[void]$lines.Add("if not all(have.values()):")
[void]$lines.Add("    print('PRISM will be downloaded from Hugging Face in Step 1b')")

$nl = [char]10
$bootstrap = ($lines -join $nl)

# --- splice into the notebook ----------------------------------------------
# -Encoding UTF8 is required here: Windows PowerShell 5.1's Get-Content
# defaults to the system codepage, not UTF-8. Without this, any non-ASCII
# character in the notebook (arrows, em dashes, the multiplication sign)
# gets misread as separate Latin-1 bytes and permanently corrupted once
# written back out - this bit us once already, and is not obvious from the
# script running without error.
$nb = Get-Content $in -Raw -Encoding UTF8 | ConvertFrom-Json

# Cell 3 is the setup cell; replace its source with the bootstrap.
$nb.cells[3].source = @($bootstrap)

$json = $nb | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($out, $json, (New-Object System.Text.UTF8Encoding($false)))

Write-Output ''
Write-Output ("wrote {0}" -f (Split-Path -Leaf $out))
$size = (Get-Item $out).Length
Write-Output ("size: {0:N0} KB" -f ($size / 1KB))
