import os
import shutil
for r, d, f in os.walk("imgs"):
    # print(r, d, f)
    for file in f:
        new_name = file[:2]
        orig = os.path.join(r, file)
        curr = os.path.join("imgs2", new_name, file)
        # print(f"{orig} => {curr}")
        os.makedirs(os.path.join("imgs2", new_name), exist_ok=True)
        shutil.move(orig, curr)