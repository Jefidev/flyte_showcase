import pandas as pd
import os

"""
SIGN_LABLES: list of labels for the 64 signs of the LSA64 dataset. 
The labels are ordered given there ID in the dataset.
"""
SIGNS_LABEL = [
    "Opaque",
    "Red",
    "Green",
    "Yellow",
    "Bright",
    "Light-blue",
    "Colors",
    "Pink",
    "Women",
    "Enemy",
    "Son",
    "Man",
    "Away",
    "Drawer",
    "Born",
    "Learn",
    "Call",
    "Skimmer",
    "Bitter",
    "Sweet milk",
    "Milk",
    "Water",
    "Food",
    "Argentina",
    "Uruguay",
    "Country",
    "Last name",
    "Where",
    "Mock",
    "Birthday",
    "Breakfast",
    "Photo",
    "Hungry",
    "Map",
    "Coin",
    "Music",
    "Ship",
    "None",
    "Name",
    "Patience",
    "Perfume",
    "Deaf",
    "Trap",
    "Rice",
    "Barbecue",
    "Candy",
    "Chewing-gum",
    "Spaghetti",
    "Yogurt",
    "Accept",
    "Thanks",
    "Shut down",
    "Appear",
    "To land",
    "Catch",
    "Help",
    "Dance",
    "Bathe",
    "Buy",
    "Copy",
    "Run",
    "Realize",
    "Give",
    "Find",
]


def write_labels(root: str):
    data = {"sign": [], "class": []}

    for idx, label in enumerate(SIGNS_LABEL):
        data["sign"].append(label)
        data["class"].append(idx)

    df = pd.DataFrame(data)
    df.to_csv(root + "/metadata/sign_to_index.csv", index=False)
