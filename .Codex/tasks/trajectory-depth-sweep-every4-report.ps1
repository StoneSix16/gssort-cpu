<#
Generate readable summary and per-frame reports from trajectory depth-sweep JSON.

Usage:
  powershell -NoProfile -File .Codex/tasks/trajectory-depth-sweep-every4-report.ps1

The script validates sampling coverage, label invariance, depth monotonicity,
and ratio denominators before writing the two Markdown reports.
#>

[CmdletBinding()]
param(
    [string[]]$InputPaths = @(
        '.Codex/tasks/trajectory-depth-sweep-every4-720.json',
        '.Codex/tasks/trajectory-depth-sweep-every4-1080.json'
    ),
    [string]$SummaryPath = '.Codex/tasks/trajectory-depth-sweep-every4-summary.md',
    [string]$FramesPath = '.Codex/tasks/trajectory-depth-sweep-every4-frame-results.md'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedFrames = @(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56)
$ExpectedScales = @(1.0, 1.1, 1.5, 2.0, 2.5, 3.0)
$Datasets = @()
$Validation = @()

function Format-Count([double]$Value) {
    return ('{0:N0}' -f $Value)
}

function Format-Pct([double]$Value) {
    return ('{0:P2}' -f $Value)
}

function Get-Variant($Frame, [double]$Scale) {
    return @($Frame.depth_variants | Where-Object {
        [Math]::Abs(([double]$_.depth_value) - $Scale) -lt 1.0e-9
    })[0]
}

function Get-FrameRatio($Metrics, [string]$Name) {
    return ([double]$Metrics.$Name / [double]$Metrics.participating)
}

function Add-Check([bool]$Passed, [string]$Description) {
    $script:Validation += [pscustomobject]@{
        passed = $Passed
        description = $Description
    }
}

foreach ($InputPath in $InputPaths) {
    if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
        throw "Input JSON not found: $InputPath"
    }
    $Data = Get-Content -Raw -LiteralPath $InputPath | ConvertFrom-Json
    $Preset = [IO.Path]::GetFileNameWithoutExtension($InputPath) -replace '^trajectory-depth-sweep-every4-', ''
    $Datasets += [pscustomobject]@{
        path = $InputPath
        preset = $Preset
        data = $Data
    }

    $Frames = @($Data.frames | ForEach-Object { [int]$_ })
    Add-Check (($Frames -join ',') -eq ($ExpectedFrames -join ',')) "$Preset samples exactly frames 0,4,...,56"
    Add-Check ($Data.frame_metrics.Count -eq $ExpectedFrames.Count) "$Preset contains 15 frame metric records"
    Add-Check ($Data.total_gaussians -eq 2131351) "$Preset uses all 2,131,351 Gaussians"
    Add-Check ($Data.fov_y_deg -eq 50) "$Preset uses FOV 50 degrees"
    Add-Check ($Data.alpha_threshold -eq (1.0 / 255.0)) "$Preset uses alpha cutoff 1/255"
    Add-Check ($Data.transmittance_threshold -eq 1.0e-4) "$Preset uses T threshold 1e-4"
    Add-Check ($Data.depth_epsilon -eq 0) "$Preset uses depth epsilon 0"
    Add-Check ($Data.pixel_sampling -eq 'centers_at_x_plus_0.5_y_plus_0.5') "$Preset uses pixel-center sampling"
    Add-Check ($Data.would_stop_definition -eq 'active_and_T_before_le_threshold') "$Preset uses T-before would-stop labels"
    Add-Check ((@($Data.depth_values | ForEach-Object {[double]$_}) -join ',') -eq ($ExpectedScales -join ',')) "$Preset contains the requested ED scales"

    foreach ($Frame in @($Data.frame_metrics)) {
        $Variants = @($Frame.depth_variants)
        Add-Check ($Variants.Count -eq $ExpectedScales.Count) "$Preset frame $($Frame.frame) contains all six scale variants"
        foreach ($Kind in @('pixel', 'fragment')) {
            $Reference = Get-Variant $Frame 1.0
            $ReferenceMetrics = $Reference.metrics.$Kind
            foreach ($Scale in $ExpectedScales) {
                $Variant = Get-Variant $Frame $Scale
                $Metrics = $Variant.metrics.$Kind
                Add-Check ($Metrics.participating -eq $ReferenceMetrics.participating) "$Preset frame $($Frame.frame) $Kind participating is scale-invariant at $Scale"
                Add-Check ($Metrics.would_stop -eq $ReferenceMetrics.would_stop) "$Preset frame $($Frame.frame) $Kind would-stop is scale-invariant at $Scale"
                $Ratio = $Variant.ratios_over_participating.$Kind
                Add-Check ([Math]::Abs(([double]$Ratio.behind_depth) - (Get-FrameRatio $Metrics 'behind_depth')) -lt 1.0e-12) "$Preset frame $($Frame.frame) $Kind behind ratio uses participating denominator at $Scale"
                Add-Check ([Math]::Abs(([double]$Ratio.would_stop) - (Get-FrameRatio $Metrics 'would_stop')) -lt 1.0e-12) "$Preset frame $($Frame.frame) $Kind would-stop ratio uses participating denominator at $Scale"
            }
            $BehindCounts = @($ExpectedScales | ForEach-Object {
                (Get-Variant $Frame $_).metrics.$Kind.behind_depth
            })
            $NonIncreasing = $true
            for ($Index = 1; $Index -lt $BehindCounts.Count; $Index++) {
                if ([int64]$BehindCounts[$Index] -gt [int64]$BehindCounts[$Index - 1]) {
                    $NonIncreasing = $false
                }
            }
            Add-Check $NonIncreasing "$Preset frame $($Frame.frame) $Kind behind-depth is non-increasing across ED scales"
        }
    }
}

function Get-Aggregate($Dataset, [double]$Scale, [string]$Kind) {
    $Rows = @($Dataset.data.frame_metrics | ForEach-Object {
        $Variant = Get-Variant $_ $Scale
        $Metrics = $Variant.metrics.$Kind
        [pscustomobject]@{
            participating = [double]$Metrics.participating
            behind_depth = [double]$Metrics.behind_depth
            would_stop = [double]$Metrics.would_stop
            behind_ratio = Get-FrameRatio $Metrics 'behind_depth'
            would_stop_ratio = Get-FrameRatio $Metrics 'would_stop'
        }
    })
    $Participating = ($Rows | Measure-Object participating -Sum).Sum
    $Behind = ($Rows | Measure-Object behind_depth -Sum).Sum
    $WouldStop = ($Rows | Measure-Object would_stop -Sum).Sum
    return [pscustomobject]@{
        participating = [int64]$Participating
        behind_depth = [int64]$Behind
        would_stop = [int64]$WouldStop
        behind_pooled = $Behind / $Participating
        would_stop_pooled = $WouldStop / $Participating
        behind_mean = ($Rows | Measure-Object behind_ratio -Average).Average
        behind_min = ($Rows | Measure-Object behind_ratio -Minimum).Minimum
        behind_max = ($Rows | Measure-Object behind_ratio -Maximum).Maximum
        would_stop_mean = ($Rows | Measure-Object would_stop_ratio -Average).Average
        would_stop_min = ($Rows | Measure-Object would_stop_ratio -Minimum).Minimum
        would_stop_max = ($Rows | Measure-Object would_stop_ratio -Maximum).Maximum
    }
}

$FailedChecks = @($Validation | Where-Object { -not $_.passed })
if ($FailedChecks.Count -gt 0) {
    $FailedChecks | ForEach-Object { Write-Error $_.description }
    throw 'Validation failed; reports were not generated.'
}

$Summary = [System.Collections.Generic.List[string]]::new()
$Summary.Add('# Trajectory depth sweep every four frames')
$Summary.Add('')
$Summary.Add('## Conditions')
$Summary.Add('')
$Summary.Add('- Source scene: `baseline.ply`; trajectory: `baseline.json` (60 frames, sampled at `0-56:4`, 15 views).')
$Summary.Add('- Project presets: 720 = 360x797; 1080 = 540x1195. These are the project dimensions, not conventional landscape 720p/1080p.')
$Summary.Add('- FOV: 50 degrees; all 2,131,351 Gaussians; ED scale values: 1.0, 1.1, 1.5, 2.0, 2.5, 3.0.')
$Summary.Add('- Thresholds: alpha `1/255`, transmittance `1e-4`, depth epsilon `0`; sampling at `(x+0.5, y+0.5)`.')
$Summary.Add('- `would_stop` means active AND `T_before <= 1e-4`. T and depth labels may overlap and must not be added.')
$Summary.Add('- Fragment labels use the existing any-qualifying-pixel semantics; they are not counts of wholly discardable fragments.')
$Summary.Add('')
$Summary.Add('Raw JSON: [720](trajectory-depth-sweep-every4-720.json), [1080](trajectory-depth-sweep-every4-1080.json).')
$Summary.Add('Per-frame readable tables: [trajectory-depth-sweep-every4-frame-results.md](trajectory-depth-sweep-every4-frame-results.md).')
$Summary.Add('')
$Summary.Add('## Validation')
$Summary.Add('')
$Summary.Add('- PASS: sample coverage, 15 records per preset, full Gaussian count, dimensions, FOV, thresholds, sampling convention, and T-before definition.')
$Summary.Add('- PASS: participating and would-stop counts are invariant across ED scales for both pair types.')
$Summary.Add('- PASS: behind-depth counts are non-increasing with ED scale for every frame and pair type.')
$Summary.Add('- PASS: every stored ratio equals its classification count divided by the corresponding participating count.')
$Summary.Add('')
$Summary.Add('## T-label totals (independent of ED scale)')
$Summary.Add('')
$Summary.Add('| Preset | Pair type | Participating | Would-stop | Pooled would-stop / participating |')
$Summary.Add('| --- | --- | ---: | ---: | ---: |')
foreach ($Dataset in $Datasets) {
    foreach ($Kind in @('pixel', 'fragment')) {
        $Aggregate = Get-Aggregate $Dataset 1.0 $Kind
        $Summary.Add(('| {0} | {1} | {2} | {3} | {4} |' -f $Dataset.preset, $Kind, (Format-Count $Aggregate.participating), (Format-Count $Aggregate.would_stop), (Format-Pct $Aggregate.would_stop_pooled)))
    }
}
$Summary.Add('')
$Summary.Add('## ED-scale aggregate totals and ratio summaries')
$Summary.Add('')
$Summary.Add('Pooled ratios are `sum(classification count) / sum(participating count)`. Frame mean/min/max are the arithmetic summaries of the 15 per-frame ratios, so they are intentionally distinct from pooled ratios.')
$Summary.Add('')
$Summary.Add('| Preset | Pair type | ED scale | Participating | Behind-depth | Would-stop | Pooled behind / P | Pooled would-stop / P | Frame behind/P mean | Frame behind/P min-max | Frame would-stop/P mean | Frame would-stop/P min-max |')
$Summary.Add('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
foreach ($Dataset in $Datasets) {
    foreach ($Kind in @('pixel', 'fragment')) {
        foreach ($Scale in $ExpectedScales) {
            $Aggregate = Get-Aggregate $Dataset $Scale $Kind
            $BehindRange = '{0} - {1}' -f (Format-Pct $Aggregate.behind_min), (Format-Pct $Aggregate.behind_max)
            $StopRange = '{0} - {1}' -f (Format-Pct $Aggregate.would_stop_min), (Format-Pct $Aggregate.would_stop_max)
            $Summary.Add(('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} |' -f $Dataset.preset, $Kind, $Scale, (Format-Count $Aggregate.participating), (Format-Count $Aggregate.behind_depth), (Format-Count $Aggregate.would_stop), (Format-Pct $Aggregate.behind_pooled), (Format-Pct $Aggregate.would_stop_pooled), (Format-Pct $Aggregate.behind_mean), $BehindRange, (Format-Pct $Aggregate.would_stop_mean), $StopRange))
        }
    }
}
$Summary.Add('')
$Summary.Add('The T labels are shown once above because they do not depend on ED scale; repeating them in every ED-scale row would imply a scale effect that is not present.')

$FrameReport = [System.Collections.Generic.List[string]]::new()
$FrameReport.Add('# Trajectory depth sweep per-frame results')
$FrameReport.Add('')
$FrameReport.Add('These tables cover frames `0,4,8,...,56` in order. Counts are raw pair counts; percentages are per-frame counts divided by that frame''s participating count.')
$FrameReport.Add('')
$FrameReport.Add('## Per-frame T labels')
$FrameReport.Add('')
foreach ($Dataset in $Datasets) {
    $FrameReport.Add(('### {0} ({1}x{2})' -f $Dataset.preset, $Dataset.data.resolution.width, $Dataset.data.resolution.height))
    $FrameReport.Add('')
    $FrameReport.Add('| Frame | Pixel participating | Pixel would-stop | Pixel W/P | Fragment participating | Fragment would-stop | Fragment W/P |')
    $FrameReport.Add('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    foreach ($Frame in @($Dataset.data.frame_metrics)) {
        $Pixel = (Get-Variant $Frame 1.0).metrics.pixel
        $Fragment = (Get-Variant $Frame 1.0).metrics.fragment
        $FrameReport.Add(('| {0} | {1} | {2} | {3} | {4} | {5} | {6} |' -f $Frame.frame, (Format-Count $Pixel.participating), (Format-Count $Pixel.would_stop), (Format-Pct (Get-FrameRatio $Pixel 'would_stop')), (Format-Count $Fragment.participating), (Format-Count $Fragment.would_stop), (Format-Pct (Get-FrameRatio $Fragment 'would_stop'))))
    }
    $FrameReport.Add('')
}
$FrameReport.Add('## Per-frame behind-depth labels by ED scale')
$FrameReport.Add('')
$FrameReport.Add('Each cell is `behind_depth (behind_depth / participating)` for the indicated pair type and scale.')
$FrameReport.Add('')
foreach ($Dataset in $Datasets) {
    $FrameReport.Add(('### {0} ({1}x{2})' -f $Dataset.preset, $Dataset.data.resolution.width, $Dataset.data.resolution.height))
    $FrameReport.Add('')
    $Header = '| Frame |'
    $Separator = '| ---: |'
    foreach ($Scale in $ExpectedScales) {
        $Header += ' s={0} pixel | s={0} fragment |' -f $Scale
        $Separator += ' ---: | ---: |'
    }
    $FrameReport.Add($Header)
    $FrameReport.Add($Separator)
    foreach ($Frame in @($Dataset.data.frame_metrics)) {
        $Line = '| {0} |' -f $Frame.frame
        foreach ($Scale in $ExpectedScales) {
            $Variant = Get-Variant $Frame $Scale
            $Pixel = $Variant.metrics.pixel
            $Fragment = $Variant.metrics.fragment
            $Line += ' {0} ({1}) | {2} ({3}) |' -f (Format-Count $Pixel.behind_depth), (Format-Pct (Get-FrameRatio $Pixel 'behind_depth')), (Format-Count $Fragment.behind_depth), (Format-Pct (Get-FrameRatio $Fragment 'behind_depth'))
        }
        $FrameReport.Add($Line)
    }
    $FrameReport.Add('')
}

$Summary -join [Environment]::NewLine | Set-Content -LiteralPath $SummaryPath -Encoding utf8
$FrameReport -join [Environment]::NewLine | Set-Content -LiteralPath $FramesPath -Encoding utf8
Write-Output "Validated $($Validation.Count) conditions."
Write-Output "Wrote $SummaryPath"
Write-Output "Wrote $FramesPath"
