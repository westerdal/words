#!/usr/bin/env python3
"""
Standalone Clue Generator
Generates OpenAI clues for word guessing games using the secret word persona approach
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))

try:
    import openai
    from scripts.utilities.config import Config
    from scripts.utilities.progress_tracker import quick_log
except ImportError:
    # Fallback for direct execution
    import openai
    from config import Config
    from progress_tracker import quick_log


def generate_clue(
    secret_word: str,
    guess_words: List[str],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict[str, Dict[str, str]]:
    """
    Generate OpenAI clues for guess words from the perspective of the secret word
    
    Args:
        secret_word: The secret word that will generate clues
        guess_words: List of words to generate clues for
        model: OpenAI model to use (default: gpt-3.5-turbo)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens for response (default: 2000)
    
    Returns:
        Dict mapping guess words to their clues and strengths
        Format: {"word": {"clue": "description", "strength": "strong/medium/weak"}}
    
    Raises:
        Exception: If OpenAI API call fails or response cannot be parsed
    """
    
    # Validate inputs
    if not secret_word or not isinstance(secret_word, str):
        raise ValueError("secret_word must be a non-empty string")
    
    if not guess_words or not isinstance(guess_words, list):
        raise ValueError("guess_words must be a non-empty list")
    
    secret_word = secret_word.lower().strip()
    guess_words = [word.lower().strip() for word in guess_words if word and isinstance(word, str)]
    
    if not guess_words:
        raise ValueError("No valid guess words provided")
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Construct the comprehensive prompt
    prompt = _build_clue_prompt(secret_word, guess_words)
    
    try:
        # Make OpenAI API call
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        clues = _parse_clue_response(content, guess_words, secret_word)
        
        return clues
        
    except Exception as e:
        raise Exception(f"OpenAI clue generation failed: {e}")


def _build_clue_prompt(secret_word: str, guess_words: List[str]) -> str:
    """Build the comprehensive clue generation prompt"""
    
    prompt = (
        f"CRITICAL: You are a '{secret_word}' speaking about guess words in a word guessing game. "
        f"For each guess word, write a riddle from YOUR PERSPECTIVE (as '{secret_word}') describing your RELATIONSHIP to that word in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: You cannot say your own name '{secret_word}' in any clue. Use 'I', 'me', 'my' instead.\n\n"
        
        f"✅ GOOD EXAMPLES (YOU are the {secret_word} speaking about guess words):\n"
        f"• Guess: 'salmon' → Clue: 'I swim in the same waters as this' (shared habitat)\n"
        f"• Guess: 'tuna' → Clue: 'I am smaller than this swimmer' (comparison)\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold than this' (comparison)\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me' (describes relationship)\n"
        f"• Guess: 'telescope' → Clue: 'I have nothing to do with this' (no relationship)\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'I am a different type of fish' ← Contains forbidden word '{secret_word}'\n"
        f"• 'I am a popular fish species' ← Contains forbidden word '{secret_word}'\n"
        f"• 'Sea creature like a fish' ← Contains forbidden word '{secret_word}'\n"
        f"• 'Type of fish that swims' ← Contains forbidden word '{secret_word}'\n\n"
        
        f"📝 MANDATORY REQUIREMENTS (You are the {secret_word}):\n"
        f"• Describe YOUR relationship to each guess word, not what the guess word is\n"
        f"• Use 'I', 'me', 'my' to refer to yourself (the {secret_word})\n"
        f"• Use 'this', 'these', 'that' to refer to the guess word\n"
        f"• Focus on connections: shared category, usage, comparison, or lack thereof\n"
        f"• For unrelated words: 'I have nothing to do with this'\n"
        f"• Connection strength: 'strong', 'medium', 'weak' (for tracking only)\n\n"
        
        f"REMEMBER: You are the {secret_word} talking about how you relate to guess words!\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {guess_words}"
    )
    
    return prompt


def _parse_clue_response(content: str, guess_words: List[str], secret_word: str) -> Dict[str, Dict[str, str]]:
    """Parse OpenAI response and extract clues"""
    
    clues = {}
    
    try:
        # Try to find JSON block in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            
            # Extract clues from parsed JSON
            for word in guess_words:
                if word in parsed_data and isinstance(parsed_data[word], dict):
                    word_data = parsed_data[word]
                    clue = word_data.get("clue", "I have no relationship to this")
                    strength = word_data.get("strength", "weak")
                    
                    # CRITICAL: Check if clue contains the secret word
                    if secret_word.lower() in clue.lower():
                        clue = "Super close, sizzling hot"
                        strength = "strong"  # These are actually very close relationships
                    
                    clues[word] = {
                        "clue": clue,
                        "strength": strength
                    }
                else:
                    # Fallback if word not found in response
                    fallback_clue = "I have no relationship to this"
                    # Check fallback clue too (shouldn't contain secret word, but safety first)
                    if secret_word.lower() in fallback_clue.lower():
                        fallback_clue = "Super close, sizzling hot"
                    clues[word] = {
                        "clue": fallback_clue,
                        "strength": "weak"
                    }
        else:
            # Fallback if no JSON found
            for word in guess_words:
                fallback_clue = "I have no relationship to this"
                if secret_word.lower() in fallback_clue.lower():
                    fallback_clue = "Super close, sizzling hot"
                clues[word] = {
                    "clue": fallback_clue,
                    "strength": "weak"
                }
                
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Fallback on parsing errors
        for word in guess_words:
            fallback_clue = "I have no relationship to this"
            if secret_word.lower() in fallback_clue.lower():
                fallback_clue = "Super close, sizzling hot"
            clues[word] = {
                "clue": fallback_clue,
                "strength": "weak"
            }
    
    return clues


def generate_single_clue(secret_word: str, guess_word: str, **kwargs) -> Tuple[str, str]:
    """
    Generate a single clue for convenience
    
    Returns:
        Tuple of (clue, strength)
    """
    result = generate_clue(secret_word, [guess_word], **kwargs)
    word_data = result.get(guess_word, {"clue": "I have no relationship to this", "strength": "weak"})
    return word_data["clue"], word_data["strength"]


def main():
    """Command line interface for standalone usage"""
    if len(sys.argv) < 3:
        print("Usage: python generate_clue.py <secret_word> <guess_word1> [guess_word2] ...")
        print("Example: python generate_clue.py fish salmon tuna telescope")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    guess_words = sys.argv[2:]
    
    print(f"🎯 Generating clues for secret word: '{secret_word}'")
    print(f"📝 Guess words: {', '.join(guess_words)}")
    print("=" * 60)
    
    try:
        clues = generate_clue(secret_word, guess_words)
        
        print(f"\n✅ Generated {len(clues)} clues:")
        for word, data in clues.items():
            print(f"  • {word}: \"{data['clue']}\" ({data['strength']})")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
