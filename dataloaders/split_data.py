import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def split_ham10000(base_dir):
    # get name->path dict
    imageid_path_dict = {}
    for x in glob(os.path.join(base_dir, '*', '*.jpg')):
        imageid_path_dict[os.path.splitext(os.path.basename(x))[0]] = x

    # This dictionary is useful for displaying more human-friendly labels later on
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    tile_df = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
    tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    train_df, test_df = train_test_split(tile_df, test_size=0.2)
    # valid_df, test_df = train_test_split(test_df, test_size=0.5)

    train_df.to_csv(os.path.join(base_dir, "train_mapping.csv"))
    # valid_df.to_csv(os.path.join(base_dir, "valid_mapping.csv"))
    test_df.to_csv(os.path.join(base_dir, "test_mapping.csv"))


if __name__ == '__main__':
    split_ham10000("/GPUFS/sysu_scjiang_2/Soraka/datasets/HAM10000")
    # df = pd.read_csv("../../input/HAM10000/train_mapping.csv")
    # test_img = Image.open(df["path"][1])
    # test_y = int(df['cell_type_idx'][1])
