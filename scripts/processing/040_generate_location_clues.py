#!/usr/bin/env python3
"""
Generate concentric rings of location clues for secret words
Creates location-based word associations with ring-based proximity clues
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

# Add utilities to path
sys.path.append(str(Path(__file__).parent.parent / "utilities"))
from config import Config
from progress_tracker import quick_log

class LocationRingGenerator:
    """Generates concentric rings of location words for any secret word"""
    
    def __init__(self, secret_word: str):
        self.secret_word = secret_word.lower().strip()
        
        # Validate word
        valid, result = Config.validate_word(self.secret_word)
        if not valid:
            raise ValueError(f"Invalid secret word: {result}")
        
        self.secret_word = result
        
        # Define the universal location hierarchy (Ring 11 -> Ring 1) - Single words only, no proper names
        self.location_rings = {
            11: ["universe", "cosmos", "multiverse", "infinity", "eternity", "void", "expanse", "totality", "existence", "reality"],
            10: ["galaxy", "system", "nebula", "constellation", "cluster", "space", "cosmos", "void", "expanse", "realm"],
            9: ["world", "globe", "planet", "sphere", "orb", "realm", "domain", "territory", "landmass", "surface"],
            8: ["continent", "hemisphere", "landmass", "ocean", "sea", "territory", "region", "expanse", "mass", "formation"],
            7: ["country", "nation", "territory", "homeland", "republic", "kingdom", "empire", "federation", "state", "dominion"],
            6: ["state", "province", "region", "county", "district", "prefecture", "canton", "territory", "division", "area"],
            5: ["city", "metropolis", "capital", "municipality", "township", "center", "hub", "locale", "settlement", "plaza"],
            4: ["town", "village", "hamlet", "settlement", "community", "borough", "township", "locality", "enclave", "district"],
            3: ["neighborhood", "district", "quarter", "suburb", "ward", "precinct", "locality", "zone", "area", "subdivision"],
            2: ["street", "avenue", "road", "lane", "boulevard", "pathway", "thoroughfare", "route", "drive", "way"]
        }
        
        # Ring 1 will be dynamically generated based on the secret word
        self.ring_1_locations = []
    
    def _determine_ring_1_locations(self) -> List[str]:
        """Determine Ring 1 locations based on the secret word's nature"""
        
        # Categories of secret words and their natural habitats (10 single-word locations each)
        location_mappings = {
            # Animals
            "animal_domestic": ["barn", "stable", "farm", "pasture", "pen", "farmyard", "paddock", "corral", "ranch", "coop"],
            "animal_wild": ["forest", "jungle", "savanna", "wilderness", "habitat", "reserve", "sanctuary", "park", "preserve", "woods"],
            "animal_aquatic": ["ocean", "river", "lake", "pond", "aquarium", "stream", "wetland", "sanctuary", "reef", "pool"],
            "animal_small": ["nest", "burrow", "den", "hole", "shelter", "hideout", "nook", "crevice", "sanctuary", "haven"],
            "animal_large": ["plains", "grassland", "territory", "reserve", "park", "savanna", "prairie", "range", "wilderness", "habitat"],
            
            # Objects/Tools
            "tool": ["workshop", "garage", "shed", "toolbox", "workbench", "basement", "storage", "shop", "studio", "facility"],
            "kitchen_item": ["kitchen", "pantry", "cupboard", "restaurant", "cafe", "galley", "cookhouse", "diner", "bistro", "eatery"],
            "furniture": ["house", "home", "room", "apartment", "mansion", "dwelling", "residence", "abode", "domicile", "quarters"],
            "book": ["library", "bookstore", "study", "school", "archive", "academy", "collection", "vault", "repository", "center"],
            "clothing": ["closet", "wardrobe", "boutique", "store", "mall", "shop", "emporium", "rack", "armoire", "outlet"],
            
            # Food
            "food": ["kitchen", "restaurant", "market", "pantry", "cafe", "diner", "bistro", "store", "delicatessen", "establishment"],
            "fruit": ["orchard", "garden", "grove", "farm", "market", "stand", "plantation", "greenhouse", "field", "plot"],
            "vegetable": ["garden", "farm", "field", "greenhouse", "market", "patch", "plot", "bed", "area", "grounds"],
            
            # Nature
            "plant": ["garden", "forest", "field", "greenhouse", "park", "arboretum", "preserve", "area", "habitat", "grove"],
            "mineral": ["mine", "cave", "quarry", "mountain", "cavern", "formation", "deposit", "outcrop", "chamber", "pit"],
            "weather": ["sky", "atmosphere", "clouds", "horizon", "heavens", "stratosphere", "dome", "zone", "layer", "realm"],
            
            # Abstract/Concepts
            "concept": ["mind", "heart", "soul", "thoughts", "dreams", "consciousness", "imagination", "realm", "world", "space"],
            "emotion": ["heart", "soul", "mind", "spirit", "being", "core", "self", "space", "center", "realm"],
            
            # Default fallback
            "default": ["home", "place", "location", "spot", "area", "dwelling", "sanctuary", "domain", "territory", "habitat"]
        }
        
        # Simple word categorization based on common patterns
        word = self.secret_word.lower()
        
        # Animal detection (basic)
        animals = ["cat", "dog", "cow", "horse", "bird", "fish", "lion", "tiger", "bear", "wolf", "deer", "rabbit", "mouse", "elephant", "whale", "shark"]
        if word in animals or any(animal in word for animal in ["fish", "bird"]):
            if word in ["fish", "shark", "whale"]:
                return location_mappings["animal_aquatic"]
            elif word in ["mouse", "rabbit", "bird"]:
                return location_mappings["animal_small"]
            elif word in ["elephant", "lion", "tiger", "bear"]:
                return location_mappings["animal_large"]
            elif word in ["cow", "horse", "pig"]:
                return location_mappings["animal_domestic"]
            else:
                return location_mappings["animal_wild"]
        
        # Tool/object detection
        tools = ["hammer", "saw", "drill", "wrench", "screwdriver", "knife", "spoon", "fork"]
        if word in tools:
            return location_mappings["tool"]
        
        # Kitchen items
        kitchen = ["pot", "pan", "plate", "cup", "bowl", "glass", "spoon", "fork", "knife"]
        if word in kitchen:
            return location_mappings["kitchen_item"]
        
        # Furniture
        furniture = ["chair", "table", "bed", "sofa", "desk", "lamp", "mirror"]
        if word in furniture:
            return location_mappings["furniture"]
        
        # Food items
        foods = ["apple", "bread", "cheese", "meat", "rice", "pasta", "cake", "cookie"]
        if word in foods:
            return location_mappings["food"]
        
        # Plants
        plants = ["tree", "flower", "grass", "rose", "oak", "pine"]
        if word in plants:
            return location_mappings["plant"]
        
        # Default: create contextual locations
        return self._create_contextual_ring_1()
    
    def _create_contextual_ring_1(self) -> List[str]:
        """Create Ring 1 locations based on word context and common sense"""
        
        # Universal Ring 1 options that work for most words
        universal_ring_1 = [
            "home", "house", "place", "location", "spot",
            "room", "space", "area", "zone", "site",
            "dwelling", "abode", "residence", "habitat", "domain",
            "sanctuary", "refuge", "haven", "quarters", "lodging"
        ]
        
        # Add word-specific contextual locations (single words only)
        contextual = []
        word = self.secret_word
        
        # If it ends in common suffixes, infer location
        if word.endswith("er") or word.endswith("or"):  # worker, actor
            contextual.extend(["office", "workplace", "building", "facility", "center", "headquarters", "establishment", "institution", "organization", "bureau"])
        elif word.endswith("ing"):  # building, reading
            contextual.extend(["hall", "center", "facility", "building", "complex", "venue", "auditorium", "chamber", "pavilion", "structure"])
        elif len(word) <= 3:  # short words
            contextual.extend(["box", "container", "holder", "case", "pocket", "compartment", "receptacle", "vessel", "storage", "enclosure"])
        else:  # longer words get general locations
            contextual.extend(["environment", "setting", "locale", "vicinity", "surroundings", "sphere", "realm", "territory", "precinct", "district"])
        
        # Combine and select exactly 10 locations
        all_options = universal_ring_1 + contextual
        unique_options = list(dict.fromkeys(all_options))  # Remove duplicates while preserving order
        return unique_options[:10]  # Return exactly 10
    
    def _generate_clue(self, guess_word: str, ring_number: int, inner_ring_locations: List[str] = None) -> str:
        """Generate a clue for a location word pointing to a specific location one ring inward"""
        
        if ring_number == 1:
            return "You have found where I live"
        
        # If no inner ring locations provided, use generic clues
        if not inner_ring_locations:
            return "Look closer to find where I dwell"
        
        # Get a unique location from the inner ring for this clue
        # Use the guess_word to deterministically select an inner location
        inner_location_index = hash(guess_word) % len(inner_ring_locations)
        target_location = inner_ring_locations[inner_location_index]
        
        # Create clue templates based on ring number, all in first person from secret word's perspective
        clue_templates = {
            2: [
                f"I live where you'll find the {target_location}",
                f"My home is near the {target_location}",
                f"Look for me by the {target_location}",
                f"I dwell close to the {target_location}",
                f"Find me where the {target_location} leads",
                f"I reside near the {target_location}",
                f"My sanctuary is by the {target_location}",
                f"Seek me where the {target_location} ends",
                f"I inhabit the area around the {target_location}",
                f"My dwelling sits beside the {target_location}"
            ],
            3: [
                f"My {target_location} runs through this area",
                f"I live near the {target_location} within here",
                f"My {target_location} is found in this locality",
                f"I dwell by the {target_location} in this place",
                f"The {target_location} where I live is within here",
                f"Seek my {target_location} in this district",
                f"My home's {target_location} lies in this area",
                f"The {target_location} near me passes through this zone",
                f"Find my {target_location} within this region",
                f"I live where the {target_location} exists here"
            ],
            4: [
                f"My {target_location} belongs to this settlement",
                f"I live in the {target_location} of this community",
                f"The {target_location} where I dwell is in this place",
                f"My home's {target_location} is within this locality",
                f"Find my {target_location} within this town",
                f"The {target_location} I inhabit is part of this area",
                f"My {target_location} is held by this community",
                f"Seek my {target_location} in this settlement",
                f"The {target_location} where I live is in this hamlet",
                f"My dwelling's {target_location} is in this place"
            ],
            5: [
                f"My {target_location} is contained within this city",
                f"The {target_location} where I live is in this metropolis",
                f"Look for my {target_location} in this urban area",
                f"My home's {target_location} is in this municipality",
                f"Find my {target_location} within this center",
                f"The {target_location} I inhabit belongs to this city",
                f"My {target_location} is held by this urban space",
                f"Seek my {target_location} in this metropolis",
                f"The {target_location} where I dwell is in this capital",
                f"My dwelling's {target_location} is in this city"
            ],
            6: [
                f"My {target_location} is contained within this region",
                f"The {target_location} where I live is in this state",
                f"Look for my {target_location} in this province",
                f"My home's {target_location} is in this territory",
                f"Find my {target_location} within this district",
                f"The {target_location} I inhabit belongs to this region",
                f"My {target_location} is held by this area",
                f"Seek my {target_location} in this county",
                f"The {target_location} where I dwell is in this division",
                f"My dwelling's {target_location} is in this province"
            ],
            7: [
                f"My {target_location} is contained within this nation",
                f"The {target_location} where I live is in this country",
                f"Look for my {target_location} in this homeland",
                f"My home's {target_location} is in this territory",
                f"Find my {target_location} within this republic",
                f"The {target_location} I inhabit belongs to this nation",
                f"My {target_location} is held by this country",
                f"Seek my {target_location} in this kingdom",
                f"The {target_location} where I dwell is in this empire",
                f"My dwelling's {target_location} is in this land"
            ],
            8: [
                f"My {target_location} is contained within this continent",
                f"The {target_location} where I live is on this landmass",
                f"Look for my {target_location} on this continent",
                f"My home's {target_location} is in this region",
                f"Find my {target_location} within this expanse",
                f"The {target_location} I inhabit belongs to this landmass",
                f"My {target_location} is held by this continent",
                f"Seek my {target_location} on this mass",
                f"The {target_location} where I dwell is on this formation",
                f"My dwelling's {target_location} is on this landmass"
            ],
            9: [
                f"My {target_location} is contained within this world",
                f"The {target_location} where I live is on this planet",
                f"Look for my {target_location} on this globe",
                f"My home's {target_location} is on this sphere",
                f"Find my {target_location} within this realm",
                f"The {target_location} I inhabit belongs to this world",
                f"My {target_location} is held by this planet",
                f"Seek my {target_location} on this orb",
                f"The {target_location} where I dwell is on this surface",
                f"My dwelling's {target_location} is on this globe"
            ],
            10: [
                f"My {target_location} is contained within this galaxy",
                f"The {target_location} where I live is in this system",
                f"Look for my {target_location} in this cluster",
                f"My home's {target_location} is in this space",
                f"Find my {target_location} within this cosmos",
                f"The {target_location} I inhabit belongs to this galaxy",
                f"My {target_location} is held by this system",
                f"Seek my {target_location} in this nebula",
                f"The {target_location} where I dwell is in this void",
                f"My dwelling's {target_location} is in this expanse"
            ],
            11: [
                f"My {target_location} is contained within this universe",
                f"The {target_location} where I live is in this cosmos",
                f"Look for my {target_location} in this reality",
                f"My home's {target_location} is in this totality",
                f"Find my {target_location} within this existence",
                f"The {target_location} I inhabit belongs to this universe",
                f"My {target_location} is held by this cosmos",
                f"Seek my {target_location} in this infinity",
                f"The {target_location} where I dwell is in this expanse",
                f"My dwelling's {target_location} is in this reality"
            ]
        }
        
        # Get templates for this ring
        templates = clue_templates.get(ring_number, [f"Look for the {target_location} within"])
        
        # Use hash to deterministically select a template based on the guess word
        template_index = hash(guess_word + str(ring_number)) % len(templates)
        return templates[template_index]
    
    def generate_location_rings(self) -> List[Tuple[str, str, int]]:
        """Generate complete location rings for the secret word"""
        
        quick_log(self.secret_word, f"🎯 Generating location rings for '{self.secret_word}'")
        
        # Determine Ring 1 locations
        self.ring_1_locations = self._determine_ring_1_locations()
        
        # Combine Ring 1 with predefined rings
        all_rings = {1: self.ring_1_locations, **self.location_rings}
        
        results = []
        used_locations = set()  # Track locations already used
        
        # Generate clues for each ring (process from inner to outer to prioritize inner rings)
        for ring_num in sorted(all_rings.keys()):
            locations = all_rings[ring_num]
            
            # Filter out locations already used in inner rings
            unique_locations = [loc for loc in locations if loc not in used_locations]
            
            quick_log(self.secret_word, f"📍 Processing Ring {ring_num}: {len(unique_locations)}/{len(locations)} unique locations")
            
            # Get inner ring locations for clue generation
            inner_ring_num = ring_num - 1
            inner_ring_locations = all_rings.get(inner_ring_num, [])
            
            for location in unique_locations:
                clue = self._generate_clue(location, ring_num, inner_ring_locations)
                results.append((location, clue, ring_num))
                used_locations.add(location)
        
        # Sort results by ring number (outermost first) for display
        results.sort(key=lambda x: x[2], reverse=True)
        
        quick_log(self.secret_word, f"✅ Generated {len(results)} location clues across {len(all_rings)} rings")
        
        return results
    
    def save_location_csv(self, output_file: Optional[str] = None) -> str:
        """Save location clues to CSV file"""
        
        if output_file is None:
            output_file = f"secretword/{self.secret_word}-location-clues.csv"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Generate the location rings
        location_data = self.generate_location_rings()
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['guess', 'clue', 'ring'])  # Header
            
            for guess, clue, ring in location_data:
                writer.writerow([guess, clue, ring])
        
        file_size = output_path.stat().st_size
        quick_log(self.secret_word, f"💾 Saved location clues to {output_path} ({file_size:,} bytes)")
        
        return str(output_path)

def generate_location_clues(secret_word: str, output_file: Optional[str] = None) -> str:
    """
    Main function to generate location clues for a secret word
    
    Args:
        secret_word: The secret word to generate location rings for
        output_file: Optional output file path (defaults to secretword/[word]-location-clues.csv)
    
    Returns:
        Path to the generated CSV file
    """
    generator = LocationRingGenerator(secret_word)
    return generator.save_location_csv(output_file)

def main():
    """Main entry point for command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python 040_generate_location_clues.py <secret_word>")
        print("Example: python 040_generate_location_clues.py cow")
        sys.exit(1)
    
    secret_word = sys.argv[1]
    
    try:
        output_file = generate_location_clues(secret_word)
        print(f"\n🎉 Successfully generated location clues for '{secret_word}'!")
        print(f"📄 Output file: {output_file}")
        
        # Show a preview of the results
        print(f"\n📋 Preview of location clues:")
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for i, row in enumerate(reader):
                if i >= 10:  # Show first 10 rows
                    print("   ... (see full file for complete results)")
                    break
                guess, clue, ring = row
                print(f"   Ring {ring}: {guess} - '{clue}'")
        
    except Exception as e:
        print(f"💥 Error generating location clues: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
