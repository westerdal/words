#!/usr/bin/env python3
"""
015_generate_word_meta.py - Generate meta files for secret words using OpenAI classification

This script generates meta JSON files containing category information for secret words.
It uses OpenAI API to classify words into predefined single-word categories.

Usage:
    python 015_generate_word_meta.py <secret_word>
    
Example:
    python 015_generate_word_meta.py cow
    
Output:
    secretword/cow-meta.json
"""

import os
import sys
import json
import openai
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utilities.progress_tracker import quick_log

class WordMetaGenerator:
    def __init__(self, secret_word: str):
        self.secret_word = secret_word
        self.openai_client = None
        self.categories = [
            "domestic", "wild", "aquatic", "tool", "kitchen", "furniture", 
            "book", "clothing", "food", "fruit", "vegetable", "plant", 
            "mineral", "weather", "concept", "emotion", "default"
        ]
        
    def setup_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            quick_log(self.secret_word, "⚠️ OPENAI_API_KEY not found in environment")
            return False
            
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            return True
        except Exception as e:
            quick_log(self.secret_word, f"❌ Failed to initialize OpenAI client: {e}")
            return False
    
    def classify_word(self) -> dict:
        """Use OpenAI to classify the word into a category"""
        if not self.setup_openai():
            return {
                "primary_category": "default",
                "reasoning": "OpenAI API not available",
                "confidence": "low",
                "error": "API setup failed"
            }
        
        prompt = f"""Classify the word '{self.secret_word}' into ONE of these single-word categories:

- domestic (pets, farm animals, domesticated creatures)
- wild (wild animals, zoo animals, untamed creatures)  
- aquatic (fish, sea creatures, water animals)
- tool (instruments, equipment, utensils)
- kitchen (cooking items, appliances, food prep)
- furniture (chairs, tables, beds, household items)
- book (literature, documents, written materials)
- clothing (garments, accessories, wearables)
- food (prepared food, dishes, meals)
- fruit (fruits and berries)
- vegetable (vegetables and herbs)
- plant (flowers, trees, plants, flora)
- mineral (rocks, metals, gems, stones)
- weather (weather phenomena, atmospheric conditions)
- concept (abstract ideas, intangible things)
- emotion (feelings, moods, emotional states)

Respond with ONLY the single word category that best fits, followed by a brief explanation.

Format your response as:
Category: [category]
Reasoning: [1-2 sentence explanation]"""

        try:
            quick_log(self.secret_word, f"🤖 Classifying word with OpenAI...")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a word classification expert. Provide accurate, single-word category classifications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the response
            lines = content.split('\n')
            category = "default"
            reasoning = "No reasoning provided"
            
            for line in lines:
                if line.startswith("Category:"):
                    category = line.replace("Category:", "").strip().lower()
                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
            
            # Validate category
            if category not in self.categories:
                quick_log(self.secret_word, f"⚠️ Invalid category '{category}', using 'default'")
                category = "default"
                reasoning = f"Invalid category returned: {category}. " + reasoning
            
            quick_log(self.secret_word, f"✅ Classified as: {category}")
            
            return {
                "primary_category": category,
                "reasoning": reasoning,
                "confidence": "high",
                "raw_response": content
            }
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ OpenAI classification failed: {e}")
            return {
                "primary_category": "default",
                "reasoning": f"OpenAI API error: {str(e)}",
                "confidence": "low",
                "error": str(e)
            }
    
    def generate_meta_file(self) -> str:
        """Generate and save the meta JSON file"""
        quick_log(self.secret_word, f"🎯 Generating meta file for '{self.secret_word}'")
        
        # Check if meta file already exists
        output_file = Path("secretword") / f"{self.secret_word}-meta.json"
        if output_file.exists():
            quick_log(self.secret_word, f"📄 Meta file already exists: {output_file}")
            return str(output_file)
        
        # Classify the word
        classification = self.classify_word()
        
        # Create meta data structure
        meta_data = {
            "secret_word": self.secret_word,
            "category": classification["primary_category"],
            "openai_classification": {
                "primary_category": classification["primary_category"],
                "reasoning": classification["reasoning"],
                "confidence": classification["confidence"]
            },
            "generated_timestamp": datetime.now().isoformat() + "Z",
            "openai_model": "gpt-3.5-turbo"
        }
        
        # Add error info if present
        if "error" in classification:
            meta_data["openai_classification"]["error"] = classification["error"]
        if "raw_response" in classification:
            meta_data["openai_classification"]["raw_response"] = classification["raw_response"]
        
        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True)
        
        # Save the meta file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
            file_size = output_file.stat().st_size
            quick_log(self.secret_word, f"💾 Saved meta file: {output_file} ({file_size:,} bytes)")
            
            return str(output_file)
            
        except Exception as e:
            quick_log(self.secret_word, f"❌ Failed to save meta file: {e}")
            raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python 015_generate_word_meta.py <secret_word>")
        print("Example: python 015_generate_word_meta.py cow")
        sys.exit(1)
    
    secret_word = sys.argv[1].strip().lower()
    
    try:
        generator = WordMetaGenerator(secret_word)
        output_file = generator.generate_meta_file()
        
        print(f"\n🎉 Successfully generated meta file for '{secret_word}'!")
        print(f"📄 Output file: {output_file}")
        
        # Show preview of the generated meta file
        with open(output_file, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        print(f"\n📋 Meta file contents:")
        print(f"   Word: {meta_data['secret_word']}")
        print(f"   Category: {meta_data['category']}")
        print(f"   Reasoning: {meta_data['openai_classification']['reasoning']}")
        print(f"   Confidence: {meta_data['openai_classification']['confidence']}")
        
    except Exception as e:
        print(f"💥 Error generating meta file for '{secret_word}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
