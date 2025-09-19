#!/usr/bin/env python3
"""
Realistic test of dynamic cutoff system using actual semantic ranking order
"""

import json
from openai import OpenAI

class RealisticCutoffTest:
    """Test cutoff system with words in realistic semantic ranking order"""
    
    def __init__(self, consecutive_weak_threshold=5):  # More realistic threshold
        self.consecutive_weak_threshold = consecutive_weak_threshold
        self.consecutive_weak_count = 0
        self.ai_cutoff_reached = False
        self.total_ai_calls = 0
        
        try:
            self.client = OpenAI()
            self.ai_available = True
        except:
            self.ai_available = False
    
    def get_relationship_strength(self, word, secret_word):
        """Get relationship strength (simplified version for testing)"""
        
        if self.ai_cutoff_reached or not self.ai_available:
            return None
        
        try:
            self.total_ai_calls += 1
            
            prompt = f"""Rate the relationship between '{word}' and '{secret_word}' as "strong", "medium", or "weak".

strong: Direct interaction (leash, collar, bone, walk, bark)
medium: Indirect connection (tree, park, house, car)  
weak: Very distant/forced (calculator, algebra, philosophy)

Just return the strength rating: strong, medium, or weak"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Lower temperature for consistency
            )
            
            strength = response.choices[0].message.content.strip().lower()
            
            # Update tracking
            if strength == "weak":
                self.consecutive_weak_count += 1
                if self.consecutive_weak_count >= self.consecutive_weak_threshold:
                    self.ai_cutoff_reached = True
                    return f"weak (CUTOFF at {self.consecutive_weak_count})"
            else:
                self.consecutive_weak_count = 0
            
            return strength
            
        except Exception as e:
            return f"error: {e}"

def main():
    """Test with realistic word progression from close to distant"""
    
    secret_word = "dog"
    
    # Simulate realistic semantic ranking progression
    # (These would normally be ranked by embedding similarity)
    test_words = [
        # Rank 1-10: Very close (should be strong)
        ("puppy", 2), ("canine", 3), ("bark", 4), ("leash", 5),
        ("collar", 6), ("bone", 7), ("walk", 8), ("fetch", 9), ("tail", 10),
        
        # Rank 100-200: Medium distance (should be medium)
        ("park", 150), ("tree", 180), ("house", 200),
        
        # Rank 1000-2000: Getting distant (should be medium->weak)
        ("animal", 1200), ("pet", 1400), ("friend", 1600),
        
        # Rank 5000+: Very distant (should be weak)
        ("car", 5200), ("book", 6800), ("music", 7500),
        ("computer", 8200), ("mathematics", 9100), ("philosophy", 9800),
        
        # Rank 15000+: Extremely distant (should definitely be weak)
        ("spoon", 15000), ("microscope", 18000), ("algebra", 20000),
        ("trigonometry", 22000), ("calculus", 24000), ("geometry", 26000),
        ("quantum", 28000), ("molecule", 30000)
    ]
    
    print("=== Realistic Dynamic Cutoff Test ===")
    print(f"Secret word: '{secret_word}'")
    print(f"Cutoff threshold: 5 consecutive weak relationships")
    print(f"Testing {len(test_words)} words across semantic ranking spectrum")
    print()
    
    cutoff_system = RealisticCutoffTest(consecutive_weak_threshold=5)
    
    for word, rank in test_words:
        print(f"Rank {rank:>5}: {word:<12} → ", end="")
        
        if cutoff_system.ai_cutoff_reached:
            print("🚫 CUTOFF (no AI call)")
            continue
        
        strength = cutoff_system.get_relationship_strength(word, secret_word)
        
        if strength:
            if "CUTOFF" in str(strength):
                print(f"💥 {strength}")
                break
            else:
                # Visual indicators
                if strength == "strong":
                    indicator = "💪"
                elif strength == "medium":  
                    indicator = "📊"
                elif strength == "weak":
                    indicator = f"⚠️  ({cutoff_system.consecutive_weak_count})"
                else:
                    indicator = "❌"
                
                print(f"{indicator} {strength}")
        else:
            print("❌ No result")
    
    print()
    print("=== Results ===")
    print(f"Total AI calls made: {cutoff_system.total_ai_calls}")
    print(f"Cutoff reached: {cutoff_system.ai_cutoff_reached}")
    
    if cutoff_system.ai_cutoff_reached:
        total_words = len(test_words)
        saved_calls = total_words - cutoff_system.total_ai_calls
        savings_percent = (saved_calls / total_words) * 100
        
        print(f"API calls saved: {saved_calls}/{total_words} ({savings_percent:.1f}%)")
        print()
        print("💡 Benefits of dynamic cutoff:")
        print("  • Automatically detects when relationships become too distant")
        print("  • Saves API costs on low-value distant words")
        print("  • Maintains quality clues for meaningful relationships")
        print("  • Scales efficiently for large word lists (100K+ words)")

if __name__ == "__main__":
    main()
