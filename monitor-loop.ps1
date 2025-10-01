#!/usr/bin/env powershell
<#
.SYNOPSIS
    Monitor the CSV-Prompt Loop Progress
#>

Write-Host "=== CSV-PROMPT LOOP MONITOR ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# Get progress stats
$embeddings = Get-ChildItem secretword\embeddings-*.txt | ForEach-Object { 
    $_.BaseName -replace 'embeddings-', '' 
} | Where-Object { 
    $_ -notmatch '2$' -and $_ -notmatch '-clean$' 
}

$csvs = Get-ChildItem secretword\secretword-*.csv | ForEach-Object { 
    if ($_.BaseName -match 'secretword-easy-animals-(.+?)(_.*)?$') { 
        $matches[1] 
    } 
}

$remaining = ($embeddings | Where-Object { $csvs -notcontains $_ }).Count
$progress = [math]::Round(($csvs.Count / $embeddings.Count) * 100, 1)

Write-Host "PROGRESS STATISTICS:" -ForegroundColor Yellow
Write-Host "  Total embeddings:   $($embeddings.Count)"
Write-Host "  Completed CSVs:     $($csvs.Count)"
Write-Host "  Words remaining:    $remaining"
Write-Host "  Progress:           $progress%"

# Simple progress bar
$barWidth = 40
$filledWidth = [math]::Round(($progress / 100) * $barWidth)
$emptyWidth = $barWidth - $filledWidth
$progressBar = "[" + ("#" * $filledWidth) + ("." * $emptyWidth) + "] $progress%"
Write-Host "  $progressBar" -ForegroundColor Green

Write-Host ""

# Check lock file
Write-Host "CURRENTLY PROCESSING:" -ForegroundColor Yellow
$lockFile = "secretword\.lock-csv.lock"
if (Test-Path $lockFile) {
    $content = Get-Content $lockFile
    $currentTime = Get-Date
    $processing = @()
    
    foreach ($line in $content) {
        if ($line -match '^(\w+)\s+(.+)$') {
            $word = $matches[1]
            $timestampStr = $matches[2]
            try {
                $timestamp = [DateTime]::Parse($timestampStr)
                $ageMinutes = [math]::Round(($currentTime - $timestamp).TotalMinutes, 1)
                $processing += "  • $word (processing for $ageMinutes minutes)"
            } catch {
                $processing += "  • $word (invalid timestamp)"
            }
        }
    }
    
    if ($processing.Count -gt 0) {
        $processing | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "  No words currently being processed" -ForegroundColor Gray
    }
} else {
    Write-Host "  No lock file found" -ForegroundColor Gray
}

Write-Host ""

# Show next few words
$needCsv = $embeddings | Where-Object { $csvs -notcontains $_ }
if ($needCsv.Count -gt 0) {
    Write-Host "NEXT WORDS IN QUEUE:" -ForegroundColor Yellow
    $next5 = $needCsv | Select-Object -First 5
    Write-Host "  $($next5 -join ', ')"
} else {
    Write-Host "🎉 ALL WORDS COMPLETED!" -ForegroundColor Green
}

Write-Host ""

# Estimate completion time
if ($remaining -gt 0 -and $csvs.Count -gt 67) {
    $wordsPerMinute = 30  # Rough estimate based on cache performance
    $estimatedMinutes = [math]::Round($remaining / $wordsPerMinute, 1)
    $estimatedHours = [math]::Round($estimatedMinutes / 60, 1)
    
    if ($estimatedHours -lt 1) {
        Write-Host "ESTIMATED COMPLETION: $estimatedMinutes minutes" -ForegroundColor Cyan
    } else {
        Write-Host "ESTIMATED COMPLETION: $estimatedHours hours" -ForegroundColor Cyan
    }
}

Write-Host "=== End Monitor Report ===" -ForegroundColor Cyan

