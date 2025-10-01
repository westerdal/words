#!/usr/bin/env python3
"""
Enhanced Embeddings Generator with OpenAI integration
Creates both embeddings-[word].txt and embeddings-[word]2.txt with OpenAI words at top
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent.parent / "utilities"))
from config import Config
from progress_tracker import create_tracker, quick_log
from 020_expand_vocabulary import get_openai_similar_words
from plural_converter import convert_plurals_to_singular
from embeddings_merger import create_merged_embeddings

# Import original embeddings generator
sys.path.append(str(Path(__file__).parent))
from create_embeddings import EmbeddingsGenerator

class EnhancedEmbeddingsGenerator(EmbeddingsGenerator):
    """Enhanced embeddings generator with OpenAI integration"""
    
    def __init__(self, secret_word: str):
        super().__init__(secret_word)
        self.openai_words = []
        self.processed_openai_words = []
    
    def get_openai_words(self) -> bool:
        """Get OpenAI similar words and process them"""
        quick_log(self.secret_word, f"🤖 Getting OpenAI similar words for '{self.secret_word}'")
        
        # Get OpenAI words (with caching) - use module directly to access update methods
        from 020_expand_vocabulary import OpenAISimilarWords
        self.openai_module = OpenAISimilarWords(self.secret_word)
        self.openai_words = self.openai_module.get_similar_words()
        
        if not self.openai_words:
            quick_log(self.secret_word, "❌ ERROR: No OpenAI words retrieved - cannot create enhanced embeddings")
            quick_log(self.secret_word, "💡 This could be due to:")
            quick_log(self.secret_word, "   - OpenAI API key not set")
            quick_log(self.secret_word, "   - API request failed")
            quick_log(self.secret_word, "   - Invalid response format")
            return False
        
        # Convert plurals to singular and remove duplicates
        self.processed_openai_words = convert_plurals_to_singular(self.secret_word, self.openai_words)
        
        if not self.processed_openai_words:
            quick_log(self.secret_word, "❌ ERROR: No words remaining after plural conversion and deduplication")
            return False
        
        # Update ENABLE2.txt with final processed words (after plural conversion)
        if hasattr(self, 'openai_module'):
            self.openai_module.update_enable2_with_final_words(self.processed_openai_words)
        
        quick_log(self.secret_word, f"✅ Processed OpenAI words: {len(self.openai_words)} → {len(self.processed_openai_words)}")
        
        return True
    
    def create_merged_embeddings_file(self) -> bool:
        """Create the enhanced embeddings2 file"""
        if not self.processed_openai_words:
            quick_log(self.secret_word, "⚠️ WARNING: No processed OpenAI words - skipping merged embeddings creation")
            return False
        
        # Create merged embeddings file
        return create_merged_embeddings(self.secret_word, self.processed_openai_words)
    
    def generate_enhanced(self) -> bool:
        """Main enhanced generation process"""
        quick_log(self.secret_word, f"🚀 Starting enhanced embeddings generation for '{self.secret_word}'")
        
        # Step 1: Get OpenAI similar words - REQUIRED for enhanced embeddings
        openai_success = self.get_openai_words()
        
        if not openai_success:
            quick_log(self.secret_word, "❌ STOPPING: Cannot create enhanced embeddings without OpenAI words")
            quick_log(self.secret_word, "💡 To create standard embeddings only, use: python scripts/embeddings/create_embeddings.py")
            return False
        
        # Step 2: Generate standard embeddings (embeddings-[word].txt)
        quick_log(self.secret_word, f"📊 Creating standard embeddings first...")
        standard_success = self.generate()
        
        if not standard_success:
            quick_log(self.secret_word, "❌ ERROR: Failed to create standard embeddings - cannot proceed")
            return False
        
        # Step 3: Create merged embeddings (embeddings-[word]2.txt)
        quick_log(self.secret_word, f"🔄 Creating enhanced embeddings with {len(self.processed_openai_words)} OpenAI words...")
        merged_success = self.create_merged_embeddings_file()
        
        if not merged_success:
            quick_log(self.secret_word, "❌ ERROR: Failed to create merged embeddings")
            return False
        
        # Success!
        quick_log(self.secret_word, f"✅ Enhanced embeddings generation completed successfully!")
        quick_log(self.secret_word, f"📁 Created files:")
        quick_log(self.secret_word, f"   - {self.paths['embeddings']} (standard embeddings)")
        quick_log(self.secret_word, f"   - {Config.SECRETWORD_DIR}/embeddings-{self.secret_word}2.txt (enhanced with OpenAI)")
        quick_log(self.secret_word, f"   - {Config.SECRETWORD_DIR}/openai-{self.secret_word}.txt (OpenAI cache)")
        
        return True

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python create_embeddings_enhanced.py <secret_word>")
        print("Example: python create_embeddings_enhanced.py forest")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    try:
        generator = EnhancedEmbeddingsGenerator(secret_word)
        success = generator.generate_enhanced()
        
        if success:
            print(f"\n🎉 Successfully generated enhanced embeddings for '{secret_word}'!")
            sys.exit(0)
        else:
            print(f"\n💥 Failed to generate enhanced embeddings for '{secret_word}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
