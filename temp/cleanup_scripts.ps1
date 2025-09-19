# PowerShell script to move older scripts to temp folder

# List of files to move to temp (older/experimental scripts)
$filesToMove = @(
    "batched_ai_generator.py",
    "complete_cat_csv.py", 
    "consolidate_embeddings.py",
    "create_cat_csv_simple.py",
    "demo_ai_clues.py",
    "dynamic_cutoff_system.py",
    "fix_dog_clues.py",
    "fix_dog_csv_queue.py",
    "generate_cat_csv.py",
    "generate_cat_dog_with_checkpoints.py",
    "incremental_cache_builder.py",
    "interactive_game_master.py",
    "optimized_generator.py",
    "process_semantic_rank_word_dynamic.py",
    "process_single_word.py",
    "pure_ai_generator.py",
    "realistic_cutoff_test.py",
    "robust_batch_generator_dynamic.py",
    "robust_batch_generator.py",
    "semantic_embedding_generator_dynamic.py",
    "test_ai_clue.py",
    "test_cached_embeddings.py",
    "test_plural_detection.py",
    "test_relationship_detection.py",
    "update_cat_dog_csvs_with_queue.py",
    "update_cat_dog_csvs.py",
    "update_csvs_with_proper_queue.py",
    "validate_final_csv.py",
    "word_game_master.py",
    "word_list.txt"
)

Write-Host "Cleaning up main directory by moving older scripts to temp/"
Write-Host ""

$moved = 0
$skipped = 0

foreach ($file in $filesToMove) {
    if (Test-Path $file) {
        try {
            Move-Item $file "temp/"
            Write-Host "Moved: $file"
            $moved++
        }
        catch {
            Write-Host "Failed to move: $file - $_"
        }
    }
    else {
        Write-Host "Skipped: $file (not found)"
        $skipped++
    }
}

Write-Host ""
Write-Host "Cleanup Summary:"
Write-Host "   Moved: $moved files"
Write-Host "   Skipped: $skipped files"
Write-Host ""
Write-Host "Cleanup complete!"
