#!/usr/bin/env powershell
<#
.SYNOPSIS
    Progress Monitor for CSV-Prompt Loop
    
.DESCRIPTION
    Monitors the progress of the automated CSV-prompt loop without interfering.
    Shows current statistics and estimated completion time.
#>

param(
    [int]$RefreshSeconds = 30,  # How often to refresh the display
    [int]$MaxChecks = 100       # Maximum number of checks before stopping
)

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
                    Timestamp = $timestamp
                }
            } catch {
                # Skip invalid entries
            }
        }
    }
    
    return $processing
}

# Initialize tracking
$startTime = Get-Date
$lastStats = $null
$checkCount = 0

Write-Host "=== CSV-Prompt Progress Monitor ==="
Write-Host "Refresh interval: $RefreshSeconds seconds"
Write-Host "Maximum checks: $MaxChecks"
Write-Host ""

while ($checkCount -lt $MaxChecks) {
    $checkCount++
    $currentTime = Get-Date
    $stats = Get-ProgressStats
    $processing = Get-CurrentlyProcessing
    
    # Clear screen and show header
    Clear-Host
    Write-Host "=== CSV-Prompt Progress Monitor ===" -ForegroundColor Cyan
    Write-Host "Check #$checkCount at $($currentTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host ""
    
    # Show main statistics
    Write-Host "📊 PROGRESS STATISTICS:" -ForegroundColor Yellow
    Write-Host "  Total embeddings files: $($stats.TotalEmbeddings)"
    Write-Host "  Completed CSV files:    $($stats.CompletedCsvs)"
    Write-Host "  Words remaining:        $($stats.Remaining)"
    Write-Host "  Progress:               $($stats.PercentComplete)% complete"
    Write-Host ""
    
    # Show progress bar
    $barWidth = 50
    $filledWidth = [math]::Round(($stats.PercentComplete / 100) * $barWidth)
    $emptyWidth = $barWidth - $filledWidth
    $progressBar = "[$('█' * $filledWidth)$('░' * $emptyWidth)] $($stats.PercentComplete)%"
    Write-Host "  $progressBar" -ForegroundColor Green
    Write-Host ""
    
    # Show currently processing
    if ($processing.Count -gt 0) {
        Write-Host "🔄 CURRENTLY PROCESSING:" -ForegroundColor Yellow
        foreach ($proc in $processing) {
            $status = if ($proc.Age -gt 60) { "⚠️ LONG RUNNING" } else { "✅ ACTIVE" }
            Write-Host "  $($proc.Word) - $($proc.Age) minutes old - $status"
        }
    } else {
        Write-Host "🔄 CURRENTLY PROCESSING: None (between tasks)" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Show next words in queue
    if ($stats.NextWords.Count -gt 0) {
        Write-Host "📋 NEXT IN QUEUE:" -ForegroundColor Yellow
        Write-Host "  $($stats.NextWords -join ', ')"
    } else {
        Write-Host "🎉 QUEUE EMPTY - ALL PROCESSING COMPLETE!" -ForegroundColor Green
        break
    }
    Write-Host ""
    
    # Calculate and show timing estimates
    if ($lastStats -and $stats.CompletedCsvs -gt $lastStats.CompletedCsvs) {
        $wordsProcessedSinceStart = $stats.CompletedCsvs
        $timeElapsed = $currentTime - $startTime
        $avgTimePerWord = $timeElapsed.TotalMinutes / $wordsProcessedSinceStart
        $estimatedTimeRemaining = $stats.Remaining * $avgTimePerWord
        
        Write-Host "⏱️  TIMING ESTIMATES:" -ForegroundColor Yellow
        Write-Host "  Elapsed time:           $([math]::Round($timeElapsed.TotalMinutes, 1)) minutes"
        Write-Host "  Average per word:       $([math]::Round($avgTimePerWord, 1)) minutes"
        Write-Host "  Estimated remaining:    $([math]::Round($estimatedTimeRemaining, 1)) minutes"
        Write-Host "  Estimated completion:   $($currentTime.AddMinutes($estimatedTimeRemaining).ToString('HH:mm:ss'))"
    }
    Write-Host ""
    
    # Check if complete
    if ($stats.Remaining -eq 0) {
        Write-Host "🎉 ALL WORDS PROCESSED! MONITORING COMPLETE!" -ForegroundColor Green
        break
    }
    
    # Show next refresh time
    $nextRefresh = $currentTime.AddSeconds($RefreshSeconds)
    Write-Host "Next refresh in $RefreshSeconds seconds at $($nextRefresh.ToString('HH:mm:ss'))..." -ForegroundColor Gray
    Write-Host "Press Ctrl+C to stop monitoring"
    
    $lastStats = $stats
    Start-Sleep -Seconds $RefreshSeconds
}

Write-Host ""
Write-Host "=== Monitoring Complete ===" -ForegroundColor Cyan

