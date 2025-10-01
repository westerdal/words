#!/usr/bin/env powershell
<#
.SYNOPSIS
    Automated CSV-Prompt Loop - Processes all remaining words until completion
    
.DESCRIPTION
    This script implements the complete csv-prompt pipeline in a loop:
    1. Scan for words needing CSV generation
    2. Lock file management with garbage collection
    3. Reserve one word for processing
    4. Execute OpenAI expansion and CSV generation
    5. Cleanup and repeat
    
    Continues until all words with embeddings have corresponding CSV files.
#>

param(
    [int]$MaxIterations = 200,  # Safety limit to prevent infinite loops
    [int]$DelaySeconds = 2      # Delay between iterations
)

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message"
}

function Get-WordsNeedingCsv {
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
    
    return $embeddings | Where-Object { $csvs -notcontains $_ } | Sort-Object
}

function Invoke-CsvPromptPipeline {
    param([string]$Word)
    
    Write-Log "Starting CSV-Prompt pipeline for word: $Word"
    
    # Step 1: Reserve the word in lock file
    $lockFile = "secretword\.lock-csv.lock"
    $currentTime = Get-Date
    $timestamp = $currentTime.ToString('yyyy-MM-ddTHH:mm:ssZ')
    $lockEntry = "$Word $timestamp"
    
    try {
        # Check if lock file exists, create if not
        if (-not (Test-Path $lockFile)) {
            @("# Lock file for CSV generation", "# Format: [word] [timestamp]") | Set-Content $lockFile
        }
        
        # Add lock entry
        $lockEntry | Add-Content $lockFile
        Write-Log "Reserved word '$Word' with timestamp $timestamp"
        
        # Step 2: Run OpenAI expansion
        Write-Log "Running OpenAI expansion for '$Word'..."
        $result = python scripts/utilities/openai_similar_words.py $Word
        if ($LASTEXITCODE -ne 0) {
            Write-Log "OpenAI expansion failed for '$Word'" "ERROR"
            return $false
        }
        Write-Log "OpenAI expansion completed successfully for '$Word'"
        
        # Step 3: Generate CSV
        Write-Log "Generating CSV for '$Word'..."
        $result = python scripts/processing/generate_csv.py $Word
        if ($LASTEXITCODE -ne 0) {
            Write-Log "CSV generation failed for '$Word'" "ERROR"
            return $false
        }
        Write-Log "CSV generation completed successfully for '$Word'"
        
        # Step 4: Verify CSV file was created
        $csvFile = "secretword\secretword-easy-animals-$Word.csv"
        if (Test-Path $csvFile) {
            $fileSize = (Get-Item $csvFile).Length
            Write-Log "✅ CSV file verified: $csvFile ($([math]::Round($fileSize/1MB, 1)) MB)"
            
            # Step 5: Remove lock entry
            $content = Get-Content $lockFile
            $updatedContent = $content | Where-Object { $_ -notmatch "^$Word\s+" }
            $updatedContent | Set-Content $lockFile
            Write-Log "Removed lock entry for '$Word'"
            
            return $true
        } else {
            Write-Log "❌ CSV file not found: $csvFile" "ERROR"
            return $false
        }
        
    } catch {
        Write-Log "Error processing '$Word': $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Clean-LockFile {
    $lockFile = "secretword\.lock-csv.lock"
    if (-not (Test-Path $lockFile)) {
        return
    }
    
    $currentTime = Get-Date
    $oneHourAgo = $currentTime.AddHours(-1)
    $content = Get-Content $lockFile
    $validLines = @()
    $cleanedCount = 0
    
    foreach ($line in $content) {
        if ($line -match '^#' -or $line -match '^\s*$') {
            $validLines += $line
            continue
        }
        
        if ($line -match '^(\w+)\s+(.+)$') {
            $word = $matches[1]
            $timestampStr = $matches[2]
            try {
                $timestamp = [DateTime]::Parse($timestampStr)
                if ($timestamp -gt $oneHourAgo) {
                    $validLines += $line
                } else {
                    $cleanedCount++
                    Write-Log "Cleaned stale lock: $word (age: $([math]::Round(($currentTime - $timestamp).TotalMinutes, 1)) minutes)"
                }
            } catch {
                $cleanedCount++
                Write-Log "Cleaned invalid lock entry: $line"
            }
        }
    }
    
    if ($cleanedCount -gt 0) {
        $validLines | Set-Content $lockFile
        Write-Log "Lock file cleaned: removed $cleanedCount stale entries"
    }
}

# Main execution loop
Write-Log "=== Starting CSV-Prompt Automated Loop ==="
Write-Log "Maximum iterations: $MaxIterations"
Write-Log "Delay between iterations: $DelaySeconds seconds"

$iteration = 0
$totalProcessed = 0
$startTime = Get-Date

while ($iteration -lt $MaxIterations) {
    $iteration++
    
    # Clean lock file first
    Clean-LockFile
    
    # Get words needing CSV generation
    $wordsNeeded = Get-WordsNeedingCsv
    $remainingCount = $wordsNeeded.Count
    
    Write-Log "=== Iteration $iteration ==="
    Write-Log "Words remaining: $remainingCount"
    
    if ($remainingCount -eq 0) {
        Write-Log "🎉 ALL WORDS PROCESSED! No more words need CSV generation." "SUCCESS"
        break
    }
    
    # Select next word to process
    $nextWord = $wordsNeeded[0]
    Write-Log "Processing word: $nextWord (position 1 of $remainingCount)"
    
    # Execute the pipeline
    $success = Invoke-CsvPromptPipeline -Word $nextWord
    
    if ($success) {
        $totalProcessed++
        Write-Log "✅ Successfully processed '$nextWord' (Total completed: $totalProcessed)"
    } else {
        Write-Log "❌ Failed to process '$nextWord'" "ERROR"
    }
    
    # Show progress
    $elapsed = (Get-Date) - $startTime
    $avgTimePerWord = if ($totalProcessed -gt 0) { $elapsed.TotalMinutes / $totalProcessed } else { 0 }
    $estimatedRemaining = if ($avgTimePerWord -gt 0 -and $remainingCount -gt 1) { 
        [math]::Round(($remainingCount - 1) * $avgTimePerWord, 1) 
    } else { 
        "Unknown" 
    }
    
    Write-Log "Progress: $totalProcessed processed, $($remainingCount - 1) remaining"
    Write-Log "Average time per word: $([math]::Round($avgTimePerWord, 1)) minutes"
    Write-Log "Estimated time remaining: $estimatedRemaining minutes"
    
    # Delay before next iteration
    if ($remainingCount -gt 1) {
        Write-Log "Waiting $DelaySeconds seconds before next iteration..."
        Start-Sleep -Seconds $DelaySeconds
    }
}

# Final summary
$endTime = Get-Date
$totalTime = $endTime - $startTime
Write-Log "=== FINAL SUMMARY ==="
Write-Log "Total iterations: $iteration"
Write-Log "Total words processed: $totalProcessed"
Write-Log "Total execution time: $([math]::Round($totalTime.TotalMinutes, 1)) minutes"
Write-Log "Average time per word: $([math]::Round($totalTime.TotalMinutes / $totalProcessed, 1)) minutes"

$finalWordsNeeded = Get-WordsNeedingCsv
if ($finalWordsNeeded.Count -eq 0) {
    Write-Log "🎉 SUCCESS: All words have been processed!" "SUCCESS"
} else {
    Write-Log "⚠️  INCOMPLETE: $($finalWordsNeeded.Count) words still need processing" "WARNING"
    Write-Log "Remaining words: $($finalWordsNeeded -join ', ')"
}

Write-Log "=== CSV-Prompt Automated Loop Complete ==="

