#!/usr/bin/env powershell
<#
.SYNOPSIS
    Simple CSV-Prompt Loop - Process all remaining words
#>

param(
    [int]$MaxIterations = 200,
    [int]$DelaySeconds = 2
)

$lockFile = "secretword\.lock-csv.lock"
$processedCount = 0
$startTime = Get-Date

Write-Host "=== STARTING AUTOMATED CSV-PROMPT LOOP ===" -ForegroundColor Green
Write-Host "Start time: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Max iterations: $MaxIterations"
Write-Host "Delay between words: $DelaySeconds seconds"
Write-Host ""

function Get-WordsNeedingCSV {
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
    
    return $embeddings | Where-Object { $csvs -notcontains $_ }
}

function Cleanup-LockFile {
    if (-not (Test-Path $lockFile)) {
        "# Lock file for CSV generation`n# Format: [word] [timestamp]" | Out-File $lockFile -Encoding UTF8
        return
    }
    
    $currentTime = Get-Date
    $validEntries = @()
    $content = Get-Content $lockFile
    
    foreach ($line in $content) {
        if ($line -match '^(\w+)\s+(.+)$') {
            $word = $matches[1]
            $timestampStr = $matches[2]
            try {
                $timestamp = [DateTime]::Parse($timestampStr)
                $ageMinutes = ($currentTime - $timestamp).TotalMinutes
                if ($ageMinutes -lt 60) {
                    $validEntries += $line
                }
            } catch {
                # Skip invalid entries
            }
        } elseif ($line -match '^#' -or $line.Trim() -eq '') {
            $validEntries += $line
        }
    }
    
    $validEntries | Out-File $lockFile -Encoding UTF8
}

# Main processing loop
for ($i = 1; $i -le $MaxIterations; $i++) {
    Write-Host "=== ITERATION $i ===" -ForegroundColor Yellow
    
    # Cleanup stale lock entries
    Cleanup-LockFile
    
    # Get words needing processing
    $needCsv = Get-WordsNeedingCSV
    
    if ($needCsv.Count -eq 0) {
        Write-Host "🎉 ALL WORDS COMPLETED!" -ForegroundColor Green
        break
    }
    
    $nextWord = $needCsv | Select-Object -First 1
    Write-Host "Processing word: $nextWord ($($needCsv.Count) remaining)"
    
    # Check if OpenAI expansion exists
    $openaiFile = "secretword\openai-$nextWord-twopass.txt"
    if (-not (Test-Path $openaiFile)) {
        Write-Host "❌ OpenAI expansion missing for '$nextWord' - skipping" -ForegroundColor Red
        continue
    }
    
    # Reserve the word
    $timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Add-Content $lockFile "$nextWord $timestamp"
    Write-Host "✅ Reserved: $nextWord"
    
    try {
        # Generate CSV
        Write-Host "🚀 Generating CSV for '$nextWord'..."
        python scripts\processing\generate_csv.py $nextWord
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Successfully generated CSV for '$nextWord'" -ForegroundColor Green
            $processedCount++
            
            # Show progress
            $totalEmbeddings = (Get-ChildItem secretword\embeddings-*.txt | Where-Object { 
                $_.BaseName -notmatch '2$' -and $_.BaseName -notmatch '-clean$' 
            }).Count
            $completedCsvs = (Get-ChildItem secretword\secretword-*.csv).Count
            $progress = [math]::Round(($completedCsvs / $totalEmbeddings) * 100, 1)
            
            Write-Host "Progress: $completedCsvs/$totalEmbeddings ($progress percent)" -ForegroundColor Cyan
        } else {
            Write-Host "❌ Failed to generate CSV for '$nextWord'" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error processing '$nextWord': $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        # Always release the word
        $content = Get-Content $lockFile
        $cleanedContent = $content | Where-Object { $_ -notmatch "^$nextWord\s+" }
        $cleanedContent | Out-File $lockFile -Encoding UTF8
        Write-Host "🔓 Released: $nextWord"
    }
    
    Write-Host ""
    
    # Small delay
    if ($DelaySeconds -gt 0) {
        Start-Sleep -Seconds $DelaySeconds
    }
}

$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "=== LOOP COMPLETED ===" -ForegroundColor Green
Write-Host "End time: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Total runtime: $($totalTime.ToString('hh\:mm\:ss'))"
Write-Host "Words processed this session: $processedCount"
Write-Host "Iterations completed: $($i-1)"

# Final status
$needCsv = Get-WordsNeedingCSV
if ($needCsv.Count -eq 0) {
    Write-Host ""
    Write-Host "🎉🎉🎉 ALL WORDS SUCCESSFULLY PROCESSED! 🎉🎉🎉" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Loop stopped with $($needCsv.Count) words remaining" -ForegroundColor Yellow
    $next5 = $needCsv | Select-Object -First 5
    Write-Host "Next words to process: $($next5 -join ', ')"
}

