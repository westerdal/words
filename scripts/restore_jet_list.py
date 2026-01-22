
target_file = r'c:\space\words\words\256-color-word-list.txt'

# The user's list from Step 144
content = "Jet Wax Gum Sap Cod Box Bag Cup Keg Jug Jar Bin Urn Lid Mat Rug Wig Fez Wok Orb Rat Fox Eel Doe Fry Ray Kit Loon Asp Bee Fly Oak Ivy Sod Dew Air Oar Fig Oat Soy Fur Pie Lip Jaw Bus Cab Rib Pit Key Awl Tag Bib Mist Mink Kale Dill Curd Blue Gold Pink Rust Sand Corn Milk Snow Plum Mint Pear Bone Jade Salt Coal Clay Wood Sage Pine Moss Fern Teal Rose Lime Bean Leaf Bark Root Seed Weed Bush Tree Fir Palm Kelp Dirt Rock Mars Wine Cork Muck Cake Tart Sash Wool Lace Jute Wire Bolt Nail Tile Pipe Pump Cog Neon Bulb Dust Dusk Soil Loam Turf Lawn Park Pond Lake Pool Hail Foam Surf Rain Dune Alp Hole Void Mold Lava Iron Zinc Lead Ruby Opal Swan Koi Dove Crow Bear Wolf Deer Lion Seal Fish Crab Frog Toad Moth Wasp Tick Worm Slug Clam Navy Gray Beet Date Rice Kiwi Brick Black White Green Brown Peach Lemon Melon Berry Apple Grape Hazel Olive Cream Wheat Grass Frost Slate Smoke Penny Sunny Poppy Kelly Butter Cherry Candy Daisy Steel Sheet Shell Pearl Amber Coral Honey Cocoa Mocha Ivory Raven Fudge Mango Robin Lilac Denim Linen Satin Twill Chain Bar Screw Drill Blade Knife Spoon Plate Glass Bread Toast Crust Dough Flour Sugar Spice Basil Yellow Purple Orange Silver Bronze Copper Nickel"

words = content.split()

print(f"Writing {len(words)} words...")

with open(target_file, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print("Done.")
