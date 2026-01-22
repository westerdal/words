# Here is the scrubbed list containing 256 distinct colors.
# Constraints Applied:
# * Count: Exactly 256 words (fits perfectly into 1 byte).
# * Length: No minimum length (short words like "Red" and "Tan" are allowed).
# * Scrubbed: Removed duplicates (e.g., kept "Gray", removed "Grey") and multi-word colors.
# * Readability: Prioritized high-imagery words (e.g., "Ruby", "Lime", "Slate") over abstract ones.
<!-- end list -->
colors = [
    # Reds & Pinks (35)
    "Red", "Ruby", "Rose", "Pink", "Rust", "Brick", "Blush", "Coral", 
    "Salmon", "Berry", "Cherry", "Maroon", "Merlot", "Garnet", "Crimson", "Scarlet", 
    "Candy", "Wine", "Blood", "Jam", "Rouge", "Apple", "Lipstick", "Currant", 
    "Azalea", "Fuchsia", "Magenta", "Punch", "Strawberry", "Cerise", "Flamingo", "Valentine",
    "Tomato", "Chili", "Poppy",

    # Oranges, Browns & Earths (45)
    "Orange", "Tan", "Beige", "Brown", "Bronze", "Copper", "Gold", "Ochre", 
    "Amber", "Ginger", "Carrot", "Pumpkin", "Clay", "Earth", "Sand", "Sepia", 
    "Sienna", "Umber", "Tawny", "Hazel", "Mocha", "Coffee", "Cocoa", "Chocolate", 
    "Walnut", "Wood", "Oak", "Cedar", "Penny", "Peach", "Apricot", "Tangerine", 
    "Marigold", "Cider", "Spice", "Tiger", "Yam", "Papaya", "Mango", "Mahogany",
    "Hickory", "Toast", "Syrup", "Caramel", "Toffee",

    # Yellows & Creams (25)
    "Yellow", "Lemon", "Citrine", "Cream", "Ivory", "Maize", "Corn", "Straw", 
    "Blonde", "Banana", "Butter", "Honey", "Brass", "Canary", "Dandelion", "Cheese", 
    "Biscuit", "Cookie", "Oat", "Linen", "Vanilla", "Bone", "Saffron", "Mustard", 
    "Wheat",

    # Greens (35)
    "Green", "Lime", "Olive", "Sage", "Mint", "Pine", "Moss", "Fern", 
    "Forest", "Jade", "Emerald", "Kelly", "Pear", "Basil", "Pickle", "Turtle", 
    "Kelp", "Clover", "Algae", "Tea", "Bamboo", "Cactus", "Grass", "Leaf", 
    "Chartreuse", "Pistachio", "Seafoam", "Shamrock", "Juniper", "Laurel", "Myrtle", "Spinach", 
    "Viridian", "Mantis", "Slime",

    # Blues & Teals (35)
    "Blue", "Cyan", "Teal", "Aqua", "Navy", "Sky", "Azure", "Cobalt", 
    "Indigo", "Denim", "Steel", "Ice", "Arctic", "Ocean", "Royal", "Spruce", 
    "Pool", "Robin", "Jay", "Marine", "Sapphire", "Lapis", "Cerulean", "Electric", 
    "Stone", "Admiral", "Cornflower", "Powder", "Turquoise", "Peacock", "Lagoon", "Glacier",
    "Aegean", "Blueberry", "Bondi",

    # Purples & Violets (25)
    "Purple", "Violet", "Lilac", "Plum", "Grape", "Orchid", "Lavender", "Amethyst", 
    "Raisin", "Heather", "Iris", "Mauve", "Fig", "Regal", "Velvet", "Pansy", 
    "Aster", "Tulip", "Lotus", "Thistle", "Mulberry", "Eggplant", "Haze", "Periwinkle",
    "Wisteria",

    # Greys, Blacks & Metals (30)
    "Gray", "Slate", "Ash", "Silver", "Smoke", "Fog", "Flint", "Iron", 
    "Lead", "Zinc", "Tin", "Chrome", "Nickel", "Pewter", "Charcoal", "Graphite", 
    "Granite", "Concrete", "Cement", "Pebble", "Black", "Jet", "Ink", "Coal", 
    "Oil", "Ebony", "Onyx", "Raven", "Soot", "Pitch",

    # Whites & Lights (26)
    "Midnight", "Shadow", "Void", "Obsidian", "White", "Snow", "Milk", "Pearl", 
    "Chalk", "Salt", "Sugar", "Rice", "Cloud", "Cotton", "Ghost", "Frost", 
    "Alabaster", "Paper", "Lily", "Daisy", "Tofu", "Porcelain", "Crystal", "Diamond",
    "Lace", "Mist"
]