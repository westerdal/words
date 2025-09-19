#!/usr/bin/env python3
"""
Dynamic AI cutoff system that stops making API calls when relationships become too weak
"""

import json
from openai import OpenAI

class DynamicAICutoffSystem:
    """System that dynamically determines when to stop making AI calls based on relationship strength"""
    
    def __init__(self, consecutive_weak_threshold=3):
        """
        Initialize the cutoff system
        
        Args:
            consecutive_weak_threshold: Number of consecutive weak relationships before stopping AI calls
        """
        self.consecutive_weak_threshold = consecutive_weak_threshold
        self.consecutive_weak_count = 0
        self.ai_cutoff_reached = False
        self.total_ai_calls = 0
        self.total_weak_relationships = 0
        
        try:
            self.client = OpenAI()
            self.ai_available = True
        except Exception as e:
            print(f"❌ OpenAI not available: {e}")
            self.ai_available = False
    
    def get_relationship_data(self, guess_word, secret_word):
        """
        Get clue and relationship strength for a word pair
        
        Returns:
            dict with 'clue', 'strength', 'ai_used' fields, or None if AI cutoff reached
        """
        
        # Check if AI cutoff has been reached
        if self.ai_cutoff_reached:
            return {
                'clue': None,  # NULL clue
                'strength': 'cutoff',
                'ai_used': False,
                'reason': f'AI cutoff reached after {self.consecutive_weak_count} consecutive weak relationships'
            }
        
        # Check if AI is available
        if not self.ai_available:
            return {
                'clue': None,
                'strength': 'no_ai',
                'ai_used': False,
                'reason': 'OpenAI not available'
            }
        
        # Make AI call
        try:
            self.total_ai_calls += 1
            
            prompt = f"""You must analyze the word '{guess_word}' and its relationship to '{secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature' instead of '{secret_word}')
2. "strength": Rate the relationship strength as "strong", "medium", or "weak"

Relationship strength guide:
- "strong": Direct interaction, clear connection (leash, collar, bone, walk)
- "medium": Indirect connection, some logical link (tree, park, house)  
- "weak": Very distant, forced connection, opposite concepts (calculator, mathematics, philosophy)

Word to analyze: {guess_word}

Example format:
{{"clue": "connects to that animal for control", "strength": "strong"}}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            clue_data = json.loads(response_content)
            
            if "clue" in clue_data and "strength" in clue_data:
                clue = clue_data["clue"]
                strength = clue_data["strength"]
                
                # Update weak relationship tracking
                if strength == "weak":
                    self.consecutive_weak_count += 1
                    self.total_weak_relationships += 1
                    
                    # Check if we've reached the cutoff threshold
                    if self.consecutive_weak_count >= self.consecutive_weak_threshold:
                        self.ai_cutoff_reached = True
                        print(f"🛑 AI Cutoff Reached! {self.consecutive_weak_count} consecutive weak relationships detected.")
                        print(f"💰 Saved future API calls. Total AI calls made: {self.total_ai_calls}")
                        
                else:
                    # Reset consecutive count if we get a non-weak relationship
                    self.consecutive_weak_count = 0
                
                return {
                    'clue': clue,
                    'strength': strength,
                    'ai_used': True,
                    'consecutive_weak': self.consecutive_weak_count,
                    'cutoff_reached': self.ai_cutoff_reached
                }
            else:
                # Fallback for malformed response
                return {
                    'clue': 'ERROR',
                    'strength': 'error',
                    'ai_used': True,
                    'reason': 'Malformed AI response'
                }
                
        except Exception as e:
            print(f"❌ AI call failed for '{guess_word}': {e}")
            return {
                'clue': 'ERROR',
                'strength': 'error',
                'ai_used': True,
                'reason': f'API error: {e}'
            }
    
    def get_stats(self):
        """Get statistics about AI usage and cutoff performance"""
        return {
            'total_ai_calls': self.total_ai_calls,
            'total_weak_relationships': self.total_weak_relationships,
            'consecutive_weak_count': self.consecutive_weak_count,
            'ai_cutoff_reached': self.ai_cutoff_reached,
            'consecutive_weak_threshold': self.consecutive_weak_threshold
        }

def test_dynamic_cutoff():
    """Test the dynamic cutoff system with a series of words"""
    
    secret_word = "dog"
    
    # Test with words that should trigger cutoff (mix strong -> very weak)
    test_words = [
        "leash",        # Strong
        "bone",         # Strong  
        "spoon",        # Weak (1) - kitchen utensil, no dog relation
        "microscope",   # Weak (2) - scientific instrument, no dog relation
        "algebra",      # Weak (3) - abstract math, should trigger cutoff
        "trigonometry", # Should not make AI call
        "calculus",     # Should not make AI call
        "geometry"      # Should not make AI call
    ]
    
    print("=== Dynamic AI Cutoff System Test ===")
    print(f"Secret word: '{secret_word}'")
    print(f"Cutoff threshold: 3 consecutive weak relationships")
    print()
    
    cutoff_system = DynamicAICutoffSystem(consecutive_weak_threshold=3)
    
    for i, word in enumerate(test_words, 1):
        print(f"{i}. Testing '{word}':")
        
        result = cutoff_system.get_relationship_data(word, secret_word)
        
        if result:
            if result['ai_used']:
                print(f"   🤖 AI Call: '{result['clue']}'")
                print(f"   💪 Strength: {result['strength']}")
                if 'consecutive_weak' in result:
                    print(f"   📊 Consecutive weak: {result['consecutive_weak']}")
                if result.get('cutoff_reached'):
                    print(f"   🛑 CUTOFF TRIGGERED!")
            else:
                print(f"   🚫 No AI Call: {result['reason']}")
                print(f"   📝 Clue: {result['clue']} (NULL)")
        
        print()
    
    # Final stats
    stats = cutoff_system.get_stats()
    print("=== Final Statistics ===")
    print(f"Total AI calls made: {stats['total_ai_calls']}")
    print(f"Total weak relationships: {stats['total_weak_relationships']}")
    print(f"AI cutoff reached: {stats['ai_cutoff_reached']}")
    
    # Calculate savings
    total_words = len(test_words)
    saved_calls = total_words - stats['total_ai_calls']
    savings_percent = (saved_calls / total_words) * 100 if total_words > 0 else 0
    
    print(f"API calls saved: {saved_calls}/{total_words} ({savings_percent:.1f}%)")
    print()
    print("💡 In a real CSV with 100,000+ words, this could save thousands of API calls!")

if __name__ == "__main__":
    test_dynamic_cutoff()
