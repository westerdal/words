
target_file = r'c:\space\words\words\256-color-word-list.txt'

# The user's list from Step 83
content = "Red Tan Ash Ice Ink Tin Sky Jay Fog Mud Sun Hay Sea Log Nut Bat Pig Egg Ham Jam Tea Oil Tar Gem Wax Gum Sap Art Box Bag Cup Mug Jug Jar Pan Pot Lid Mat Rug Hat Cap Pen Web Rat Fox Ape Elk Cub Pup Kit Owl Ant Bee Fly Oak Ivy Sod Dew Air Gas Fig Oat Rye Fur Pie Lip Eye Bus Cab Rib Dot Key Saw Bow Tie Mist Mink Kale Dill Yolk Blue Gold Pink Rust Sand Corn Milk Snow Plum Mint Pear Bone Jade Salt Coal Clay Wood Sage Pine Moss Fern Teal Rose Lime Bean Leaf Bark Root Seed Weed Bush Tree Fir Palm Kelp Dirt Rock Mars Wine Cork Silt Cake Tart Silk Wool Lace Rope Wire Bolt Nail Tile Pipe Pump Gear Lamp Bulb Dust Grit Soil Peat Turf Lawn Park Pond Lake Pool Wave Foam Surf Tide Dune Peak Cave Mine Mold Lava Iron Zinc Lead Ruby Opal Swan Duck Dove Crow Bear Wolf Deer Lion Seal Fish Crab Frog Toad Moth Wasp Gnat Worm Slug Clam Navy Gray Beet Date Rice Kiwi Brick Black White Green Brown Peach Lemon Melon Berry Apple Grape Hazel Olive Cream Wheat Grass Frost Slate Smoke Penny Sunny Poppy Kelly Butter Cherry Candy Daisy Steel Sheet Shell Pearl Amber Coral Honey Cocoa Mocha Ivory Raven Fudge Mango Robin Lilac Denim Linen Satin Fleece Chain Valve Screw Drill Blade Knife Spoon Plate Glass Bread Toast Crust Dough Flour Sugar Spice Basil Yellow Purple Orange Silver Bronze Copper Nickel"

words = content.split()

print(f"Writing {len(words)} words...")

with open(target_file, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print("Done.")
