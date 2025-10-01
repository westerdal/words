#!/usr/bin/env powershell
# Simple Progress Monitor for CSV-Prompt Loop

function Get-ProgressStats {
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
    
    $needCsv = $embeddings | Where-Object { $csvs -notcontains $_ } | Sort-Object
    
    return @{
        TotalEmbeddings = $embeddings.Count
        CompletedCsvs = $csvs.Count
        Remaining = $needCsv.Count
        PercentComplete = [math]::Round(($csvs.Count / $embeddings.Count) * 100, 1)
        NextWords = $needCsv | Select-Object -First 5
    }
}

function Get-CurrentlyProcessing {
    $lockFile = "secretword\.lock-csv.lock"
    if (-not (Test-Path $lockFile)) {
        return @()
    }
    
    $currentTime = Get-Date
    $content = Get-Content $lockFile
    $processing = @()
    
    foreach ($line in $content) {
        if ($line -match '^(\w+)\s+(.+)$') {
            $word = $matches[1]
            $timestampStr = $matches[2]
            try {
                $timestamp = [DateTime]::Parse($timestampStr)
                $ageMinutes = [math]::Round(($currentTime - $timestamp).TotalMinutes, 1)
                $processing += @{
                    Word = $word
                    Age = $ageMinutes
                }
            } catch {
                # Skip invalid entries
            }
        }
    }
    
    return $processing
}

# Get current status
$currentTime = Get-Date
$stats = Get-ProgressStats
$processing = Get-CurrentlyProcessing

Write-Host "=== CSV-Prompt Progress Status ===" -ForegroundColor Cyan
Write-Host "Time: $($currentTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host ""

Write-Host "PROGRESS STATISTICS:" -ForegroundColor Yellow
Write-Host "  Total embeddings files: $($stats.TotalEmbeddings)"
Write-Host "  Completed CSV files:    $($stats.CompletedCsvs)"
Write-Host "  Words remaining:        $($stats.Remaining)"
Write-Host "  Progress:               $($stats.PercentComplete)% complete"
Write-Host ""

# Simple progress bar
$barWidth = 40
$filledWidth = [math]::Round(($stats.PercentComplete / 100) * $barWidth)
$emptyWidth = $barWidth - $filledWidth
$progressBar = "[" + ("#" * $filledWidth) + ("." * $emptyWidth) + "] $($stats.PercentComplete)%"
Write-Host "  $progressBar" -ForegroundColor Green
Write-Host ""

if ($processing.Count -gt 0) {
    Write-Host "CURRENTLY PROCESSING:" -ForegroundColor Yellow
    foreach ($proc in $processing) {
        $status = if ($proc.Age -gt 60) { "(LONG RUNNING)" } else { "(ACTIVE)" }
        Write-Host "  $($proc.Word) - $($proc.Age) minutes old $status"
    }
} else {
    Write-Host "CURRENTLY PROCESSING: None (between tasks)" -ForegroundColor Gray
}
Write-Host ""

if ($stats.NextWords.Count -gt 0) {
    Write-Host "NEXT IN QUEUE:" -ForegroundColor Yellow
    Write-Host "  $($stats.NextWords -join ', ')"
} else {
    Write-Host "QUEUE EMPTY - ALL PROCESSING COMPLETE!" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== End Status Report ===" -ForegroundColor Cyan

