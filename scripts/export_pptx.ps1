param([string]$Pptx = "figures\workflow.pptx", [string]$OutPng = "figures\workflow.png", [string]$OutPdf = "figures\workflow.pdf", [int]$Width = 3000)
$ErrorActionPreference = "Stop"
$pp = New-Object -ComObject PowerPoint.Application
$full = (Resolve-Path $Pptx).Path
$pres = $pp.Presentations.Open($full, $true, $false, $false)   # ReadOnly, Untitled, WithWindow=false
$wpx = $Width; $hpx = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
$png = Join-Path (Get-Location).Path $OutPng; $pdf = Join-Path (Get-Location).Path $OutPdf
$pres.Slides.Item(1).Export($png, "PNG", $wpx, $hpx)
$pres.SaveAs($pdf, 32)   # ppSaveAsPDF
$pres.Close(); $pp.Quit()
"exported $png and $pdf"
