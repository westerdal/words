#!/usr/bin/env powershell
<#
.SYNOPSIS
    Multi-Process Manager for CSV-Prompt Loops
    
.DESCRIPTION
    Helps start, stop, and monitor multiple CSV-prompt processing loops
#>

param(
    [string]$Action = "status",  # status, start, stop, restart
    [int]$ProcessCount = 2       # How many additional processes to start
)

function Get-CsvProcesses {
    return Get-Process powershell | Where-Object { 
        $_.StartTime -gt (Get-Date).AddHours(-24) -and
        $_.ProcessName -eq "powershell"
    }
}

function Get-ProcessingStats {
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
    
    return @{
        Total = $embeddings.Count
        Completed = $csvs.Count
        Remaining = ($embeddings | Where-Object { $csvs -notcontains $_ }).Count
        Progress = [math]::Round(($csvs.Count / $embeddings.Count) * 100, 1)
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

switch ($Action.ToLower()) {
    "start" {
        Write-Host "=== STARTING $ProcessCount ADDITIONAL PROCESSES ===" -ForegroundColor Green
        Write-Host ""
        
        for ($i = 1; $i -le $ProcessCount; $i++) {
            $delay = 2 + $i  # Stagger start times
            Write-Host "Starting Process #$i (delay: $delay seconds)..."
            Start-Process powershell -ArgumentList @(
                "-ExecutionPolicy", "Bypass", 
                "-File", "csv-prompt-loop.ps1", 
                "-MaxIterations", "200", 
                "-DelaySeconds", "$delay"
            ) -WindowStyle Minimized
            Start-Sleep 1
        }
        
        Write-Host ""
        Write-Host "✅ Started $ProcessCount additional processes!" -ForegroundColor Green
        Write-Host "Use 'manage-processes.ps1 status' to monitor progress"
    }
    
    "stop" {
        Write-Host "=== STOPPING ALL CSV-PROMPT PROCESSES ===" -ForegroundColor Red
        Write-Host ""
        
        $processes = Get-CsvProcesses
        foreach ($proc in $processes) {
            Write-Host "Stopping Process ID: $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        
        Write-Host "✅ Stopped all processes" -ForegroundColor Green
    }
    
    "restart" {
        Write-Host "=== RESTARTING PROCESSES ===" -ForegroundColor Yellow
        & $PSCommandPath -Action stop
        Start-Sleep 3
        & $PSCommandPath -Action start -ProcessCount $ProcessCount
    }
    
    "status" {
        Write-Host "=== MULTI-PROCESS CSV-PROMPT STATUS ===" -ForegroundColor Cyan
        Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host ""
        
        # Show all processes
        Write-Host "ACTIVE PROCESSES:" -ForegroundColor Yellow
        $processes = Get-CsvProcesses
        if ($processes.Count -gt 0) {
            $processes | Select-Object Id, ProcessName, 
                @{Name='Runtime(min)';Expression={[math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1)}}, 
                @{Name='CPU';Expression={$_.CPU}} | Format-Table
        } else {
            Write-Host "  No active processes found" -ForegroundColor Gray
        }
        
        # Show currently processing words
        Write-Host "CURRENTLY PROCESSING:" -ForegroundColor Yellow
        $processing = Get-CurrentlyProcessing
        if ($processing.Count -gt 0) {
            foreach ($proc in $processing) {
                $status = if ($proc.Age -gt 60) { "(LONG RUNNING)" } else { "(ACTIVE)" }
                Write-Host "  • $($proc.Word) - $($proc.Age) minutes $status"
            }
        } else {
            Write-Host "  No words currently being processed" -ForegroundColor Gray
        }
        Write-Host ""
        
        # Show overall progress
        $stats = Get-ProcessingStats
        Write-Host "OVERALL PROGRESS:" -ForegroundColor Yellow
        Write-Host "  • Total embeddings: $($stats.Total)"
        Write-Host "  • Completed CSVs: $($stats.Completed)"
        Write-Host "  • Remaining: $($stats.Remaining)"
        Write-Host "  • Progress: $($stats.Progress)%"
        
        # Simple progress bar
        $barWidth = 40
        $filledWidth = [math]::Round(($stats.Progress / 100) * $barWidth)
        $emptyWidth = $barWidth - $filledWidth
        $progressBar = "[" + ("#" * $filledWidth) + ("." * $emptyWidth) + "] $($stats.Progress)%"
        Write-Host "  $progressBar" -ForegroundColor Green
        
        # Estimate completion time
        if ($processing.Count -gt 0 -and $stats.Remaining -gt 0) {
            $avgTimePerWord = 15  # minutes (rough estimate)
            $parallelFactor = [math]::Min($processing.Count, 3)  # Max 3 effective parallel processes
            $estimatedHours = [math]::Round(($stats.Remaining * $avgTimePerWord) / (60 * $parallelFactor), 1)
            Write-Host "  • Estimated completion: $estimatedHours hours (with $parallelFactor parallel processes)" -ForegroundColor Cyan
        }
    }
    
    default {
        Write-Host "=== CSV-PROMPT PROCESS MANAGER ===" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: manage-processes.ps1 [Action] [ProcessCount]"
        Write-Host ""
        Write-Host "Actions:"
        Write-Host "  status   - Show current status (default)"
        Write-Host "  start    - Start additional processes"
        Write-Host "  stop     - Stop all processes"
        Write-Host "  restart  - Restart all processes"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  manage-processes.ps1 status"
        Write-Host "  manage-processes.ps1 start 2"
        Write-Host "  manage-processes.ps1 stop"
        Write-Host "  manage-processes.ps1 restart 3"
    }
}

