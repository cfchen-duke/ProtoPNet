import shutil
import pandas as pd
import os

df = pd.read_csv("meta.csv", index_col='case_num')
new_df = pd.DataFrame(index=df.index)

feat = {
    'bwv': 'blue_whitish_veil',
    'dag': 'dots_and_globules',
    'pig': 'pigmentation',
    'pn': 'pigment_network',
    'rs': 'regression_structures',
    'str': 'streaks',
}

sub_feat = {
    'bwv': ['absent', 'present'],
    'dag': ['absent', 'irregular', 'regular'],
    'pig': ['absent', 'irregular', 'regular'],
    'pn': ['absent', 'atypical', 'typical'],
    'rs': ['absent', 'present'],
    'str': ['absent', 'irregular', 'regular']
}


def create_csv(feat, sub_feat, new_df):
    new_df['img'] = df['derm'].str.split('/').str[1]
    for k, v in feat.items():
        for kk, vv in sub_feat.items():
            if k == kk:
                for i in vv:
                    col = k + '_' + i
                    new_df[col] = 0

    for index, row in df.iterrows():
        # id = row.index()
        for k, v in feat.items():
            for kk, vv in sub_feat.items():
                if k == kk:
                    for i in vv:
                        if df.at[index, v] == i:
                            new_df.at[index, kk + '_' + i] = 1

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    new_df.to_csv("one_hot_dataset.csv", index=False, header=True)


def create_dataset():
    from_dir = '../derm7pt/images/'
    to_dir = '../images/'
    df = pd.read_csv("one_hot_dataset.csv")
    c=0
    for root, _, files in os.walk(from_dir):
        for file in files:
            if file in df.img.unique():
                c+=1
                from_d = os.path.join(root,file)
                to_d = os.path.join(to_dir, file)
                shutil.copy2(from_d,to_d)
    print(c)


create_dataset()
