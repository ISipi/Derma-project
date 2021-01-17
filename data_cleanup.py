#import tensorflow as tf
import pandas as pd
import openpyxl
import os
import numpy as np

data_loc = r"E:\Dermatology datasets\PH2Dataset\PH2 Dataset images"


"""
Data from: https://www.fc.up.pt/addi/ph2%20database.html

This script loops through the data_loc provided above, looks for a folder named "roi" (ignores it) and proceeds to 
look for files with "lesion" in their name. These are the PH2 dataset's segments, while the other image included is the
ground truth. 'roi' images are ignored as they are binary masks of colour classes presented in the skin lesions and 
provide too much fine detail for the current use. We then create a dictionary of file names from the acquired files.

In the next step, the provided excel spreadsheet is opened as Pandas dataframe and cleaned. The dictionary of file names
that we created above is converted into a Pandas dataframe as well and the two dataframes are merged based on the image 
name column.

From here, we drop columns that we don't care about and make a single 'Diagnosis' column, where each image's clinical
diagnosis is inserted. Now we have labels for each original image and their segment that we can use to train a neural 
network.

Time taken: 45-60 minutes.
New libraries: openpyxl required for opening xlsx in Pandas.

"""

# create a dictionary of file names
img_dict = {}
for folders in os.listdir(data_loc):
    gt_n_lesion = []  # ground truth and lesion images as a list, which is sent to img_dict as an entry
    for folder in os.listdir(os.path.join(data_loc, folders)):
        if "roi" in folder:
            break
        else:
            for img in os.listdir(os.path.join(os.path.join(data_loc, folders), folder)):
                if "lesion" not in img:
                    gt_n_lesion.append(os.path.join(os.path.join(os.path.join(data_loc, folders), folder), img))
                elif "lesion" in img:
                    assert len(gt_n_lesion) == 1
                    gt_n_lesion.insert(1, os.path.join(os.path.join(os.path.join(data_loc, folders), folder), img))

    img_dict[folders] = tuple(gt_n_lesion)


# Set some Pandas options
pd.options.display.max_colwidth = 200
pd.set_option('display.max_columns', None)

# Data clean up and merging image locations dictionary with the provided excel sheet
new_path = os.path.split(data_loc)[0]
data_txt = os.path.join(new_path, 'PH2_dataset.xlsx')
df = pd.read_excel(str(data_txt))
df = df[df["Unnamed: 0"].notna()]
df = df.rename(columns=df.iloc[0])
df = df.drop(df.index[0])
df = df.set_index("Image Name")

img_name_df = pd.DataFrame.from_dict(img_dict, orient='index', columns=["Ground truth", "Segment"])
img_name_df.index.name = "Image Name"

merged_df = df.merge(img_name_df, on="Image Name")

# Select the rows where there are values for these columns and create a single column based on them
conditions = [(merged_df['Common Nevus'].notna()), (merged_df['Atypical Nevus'].notna()), (merged_df['Melanoma'].notna())]
values = ['Common Nevus', 'Atypical Nevus', 'Melanoma']
merged_df['Diagnosis'] = np.select(conditions, values)

final_df = merged_df.drop(['Histological Diagnosis', 'Common Nevus', 'Atypical Nevus', 'Melanoma', 'Asymmetry\n(0/1/2)',
                          'Pigment Network\n(AT/T)', 'Dots/Globules\n(A/AT/T)', 'Streaks\n(A/P)',
                          'Regression Areas\n(A/P)', 'Blue-Whitish Veil\n(A/P)', 'White', 'Red', 'Light-Brown',
                          'Dark-Brown', 'Blue-Gray', 'Black'], axis=1)



final_df.to_csv("derma_data.csv")
