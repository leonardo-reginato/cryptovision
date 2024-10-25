import os
import pandas as pd

def create_image_dataframe(image_path):
    label = []
    path = []

    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            if filename.startswith('.'):
                continue  # Ignore files starting with a dot
            if os.path.splitext(filename)[1] in (".jpeg", ".png", ".jpg"):
                if dirname.split()[-1] != "GT":
                    label.append(os.path.split(dirname)[1])
                    path.append(os.path.join(dirname, filename))

    df_og = pd.DataFrame(columns=["path", "label"])
    df_og["path"] = path
    df_og["label"] = label
    df_og["label"] = df_og["label"].astype("category")

    # Split the 'label' column into 'family', 'genus', and 'species' columns
    df_og[["family", "genus", "species"]] = df_og["label"].str.split("_", expand=True)

    return df_og