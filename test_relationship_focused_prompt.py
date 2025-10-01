#!/usr/bin/env python3
"""
Test the relationship-focused prompt where fish describes its relationships
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_relationship_focused_prompt():
    """Test the prompt focused on relationships from fish's perspective"""
    secret_word = "fish"
    
    # Test with mix of related and unrelated words
    test_words = ["starfish", "fishing", "telescope", "democracy", "goldfish", "pizza", "ocean", "bicycle", "hook", "computer"]
    
    print(f"🧪 Testing RELATIONSHIP-FOCUSED prompt")
    print(f"🐟 You are a '{secret_word}' describing your relationships")
    print(f"📝 Test words: {test_words}")
    print("=" * 70)
    
    # Use the updated relationship-focused prompt
    prompt = (
        f"CRITICAL: You are a '{secret_word}' speaking about guess words in a word guessing game. "
        f"For each guess word, write a riddle from YOUR PERSPECTIVE (as '{secret_word}') describing your RELATIONSHIP to that word in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: You cannot say your own name '{secret_word}' in any clue. Use 'I', 'me', 'my' instead.\n\n"
        
        f"✅ GOOD EXAMPLES (YOU are the {secret_word} speaking about guess words):\n"
        f"• Guess: 'starfish' → Clue: 'I am a type of these' (you share category)\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me' (describes relationship)\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold' (comparison)\n"
        f"• Guess: 'telescope' → Clue: 'I have nothing to do with this' (no relationship)\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'I see far into the sky' ← Wrong perspective (telescope describing itself)\n"
        f"• 'People choose me to govern' ← Wrong perspective (democracy describing itself)\n"
        f"• 'Sea creature like a fish' ← Contains forbidden word '{secret_word}'\n"
        f"• 'Type of fish that swims' ← Contains forbidden word '{secret_word}'\n\n"
        
        f"📝 MANDATORY REQUIREMENTS (You are the {secret_word}):\n"
        f"• Describe YOUR relationship to each guess word, not what the guess word is\n"
        f"• Use 'I', 'me', 'my' to refer to yourself (the {secret_word})\n"
        f"• Use 'this', 'these', 'that' to refer to the guess word\n"
        f"• Focus on connections: shared category, usage, comparison, or lack thereof\n"
        f"• For unrelated words: 'I have nothing to do with this'\n"
        f"• Connection strength: 'strong', 'medium', 'weak'\n\n"
        
        f"REMEMBER: You are the {secret_word} talking about how you relate to guess words!\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {test_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Response: {len(content)} characters")
        
        try:
            results = json.loads(content)
            print(f"\n✅ RESULTS ({len(results)} words processed):")
            print("-" * 70)
            
            violations = []
            wrong_perspective = []
            good_relationships = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                
                # Check if it's describing the relationship (fish's perspective)
                is_relationship = any(phrase in clue.lower() for phrase in [
                    'i am', 'i have', 'i live', 'used for', 'nothing to do', 'unrelated to me',
                    'similar to me', 'different from me', 'part of me', 'related to me'
                ])
                
                # Status indicators
                if violation:
                    status = "❌ VIOLATION"
                elif not is_relationship:
                    status = "⚠️ WRONG PERSPECTIVE"
                else:
                    status = "✅ GOOD RELATIONSHIP"
                
                print(f"{status} | {word:12} | {clue:35} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                elif not is_relationship:
                    wrong_perspective.append(f"'{word}' → '{clue}'")
                else:
                    good_relationships.append(f"'{word}' → '{clue}'")
            
            print("-" * 70)
            print(f"📊 ANALYSIS:")
            print(f"   Words processed: {len(results)}")
            print(f"   ❌ Secret word violations: {len(violations)}")
            print(f"   ⚠️ Wrong perspective: {len(wrong_perspective)}")
            print(f"   ✅ Good relationships: {len(good_relationships)}")
            
            relationship_rate = len(good_relationships) / len(results) * 100 if results else 0
            print(f"   🎯 Relationship success: {relationship_rate:.1f}%")
            
            if violations:
                print(f"\n🚨 SECRET WORD VIOLATIONS:")
                for violation in violations:
                    print(f"   • {violation}")
            
            if wrong_perspective:
                print(f"\n⚠️ WRONG PERSPECTIVE (describing guess word, not relationship):")
                for wrong in wrong_perspective[:5]:
                    print(f"   • {wrong}")
            
            if good_relationships:
                print(f"\n🎉 GOOD RELATIONSHIPS:")
                for good in good_relationships[:5]:
                    print(f"   • {good}")
            
            if relationship_rate >= 80:
                print(f"\n🏆 EXCELLENT! Fish is properly describing relationships!")
            elif relationship_rate >= 60:
                print(f"\n👍 GOOD! Most are relationship-focused!")
            else:
                print(f"\n⚠️ NEEDS WORK! Still too many wrong perspectives!")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error: {e}")
            print(f"Raw: {content}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_relationship_focused_prompt()


