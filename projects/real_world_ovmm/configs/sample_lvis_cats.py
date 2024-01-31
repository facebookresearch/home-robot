import json
import random

with open("lvis_categories.json") as f:
    lvis_categories = json.load(f)
random.shuffle(lvis_categories)
current_map = json.load(open("example_cat_map.json"))
count = 0
for cat in lvis_categories:
    if count >= 800:
        break
    if (
        cat["name"] not in current_map["obj_category_to_obj_category_id"]
        and cat["name"] not in current_map["recep_category_to_recep_category_id"]
    ):
        current_map["obj_category_to_obj_category_id"][cat["name"]] = len(
            current_map["obj_category_to_obj_category_id"]
        )
        count += 1
with open("lvis_cat_map.json", "w") as f:
    json.dump(current_map, f, indent=4)
