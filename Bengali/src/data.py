import pyarrow.parquet as pq
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#137x236

WIDTH = 137
HEIGHT = 236


def read_data(path):
    print('reading...', path)
    ext = path.split('.')[-1]
    assert(ext in ['csv', 'parquet'])
    if ext == 'csv':
        df = pd.read_csv(path)
    else:
        df = pq.read_table(path).to_pandas()
    print('columns:', df.columns)
    return df


def to_img(row):
    return np.array(row['0':'32331'].values.reshape((WIDTH, HEIGHT)), np.uint8)


def get_component(type, label, df):
    return df.loc[(df['component_type'] == type) & (df['label'] == label)]['component'].values


def view_img_by_components(components, values, df, img_df, class_map=None, size=5):
    assert(len(components) == len(values))

    if class_map is not None:
        for i in range(len(components)):
            print('{} is {}'.format(components[i], get_component(components[i], values[i], class_map)))

    mask = True
    for i in range(len(components)):
        mask &= df[components[i]].isin([values[i]])
    ids = df[mask].head(size * size)['image_id'].values
    imgs = img_df[img_df['image_id'].isin(ids)]
    fig, ax = plt.subplots(size, size, figsize=(20, 20))
    for i, index in enumerate(imgs.index):
        ax[i // size, i % size].imshow(to_img(imgs.iloc[i]))
        ax[i // size, i % size].set_title(index)
    plt.show()


def plot_distribution(feature, title, df, size=20):
    f, ax = plt.subplots(1, 1, figsize=(100, 20))
    total = float(len(df))
    g = sns.countplot(df[feature], order=df[feature].value_counts().index[:size], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 3,
                '{:1.2f}%'.format(100 * height / total),
                ha="center")
    plt.show()


def plot_couple_heatmap(f1, f2, df):
    tmp = df.groupby([f1, f2])['image_id'].count()
    df_m = tmp.reset_index()
    f1, f2 = (f1, f2) if df_m[f1].nunique() < df_m[f2].nunique() else (f2, f1)
    df_m = df_m.pivot(f1, f2, "image_id")
    f, ax = plt.subplots(figsize=(150, 20))
    sns.heatmap(df_m, annot=True, fmt='3.0f', linewidths=.5, ax=ax)
    plt.show()


if __name__ == '__main__':
    train_df = read_data('../data/train.csv')
    train_img_df = read_data('../data/train_image_data_0.parquet')
    class_map = read_data('../data/class_map.csv')
    view_img_by_components(['vowel_diacritic', 'grapheme_root'], [0, 0], train_df, train_img_df, class_map=class_map)
    #plot_couple_heatmap('vowel_diacritic', 'consonant_diacritic', train_df)
    #plot_couple_heatmap('grapheme_root', 'consonant_diacritic', train_df)
    #plot_couple_heatmap('vowel_diacritic', 'grapheme_root', train_df)

#    plot_couple_heatmap('vowel_diacritic', 'consonant_diacritic', train_df)
#    for col in train_df.columns[1:]:
#        plot_distribution(col, '{} (most frequent values in train)'.format(col), train_df)