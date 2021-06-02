import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score as mmi

COL_N_ERROR = 'new_error_types'
COL_LANG = 'learner_nationality'
COL_LEVEL = 'level'


def print_to_log(*text, new_path=None):
    LOG_PATH = 'vis_log'

    import datetime
    if new_path:
        # global LOG_PATH
        LOG_PATH = new_path
    with open(LOG_PATH, 'a') as f:
        for t in text:
            f.write("".join(str(datetime.datetime.now()).split('.')[:-1])+' - ')
            f.write(str(t))
            f.write('\n')
    print(text)


def compare_dimitry(df, dimitry_path, lang):
    dm = pd.read_csv(dimitry_path)
    dm.index = dm.en
    dm.drop(labels=['en'], inplace=True, axis=1)
    dm.drop(labels=['INTJ'], inplace=True)  # 16X16

    ours = get_heat(df[df[COL_LANG] == lang])
    # TODO: what needs to be dropped for each language.
    ours.drop(labels=['PUNCT'], inplace=True, axis=1)
    ours.drop(labels=['PUNCT'], inplace=True)
    ours.drop(labels=['INTJ'], inplace=True, axis=1)
    ours.drop(labels=['INTJ'], inplace=True)
    if lang == 'fr':
        dm.drop(labels=['PART'], inplace=True)  # 16X16
        ours.drop(labels=['PART'], inplace=True, axis=1)
        ours.drop(labels=['PART'], inplace=True)
    if lang == 'jp':
        dm.drop(labels=['PUNCT'], inplace=True)
        dm.drop(labels=['X'], inplace=True)
        ours.drop(labels=['X'], inplace=True)
        ours.drop(labels=['X'], inplace=True, axis=1)
    if lang == 'cn':
        dm.drop(labels=['SYM'], inplace=True)
    # kl(dm.values.reshape((-1,)), ours.values.reshape((-1,)))
    return mmi(dm.values.reshape((-1,)), ours.values.reshape((-1,))) # or avg score for every line



def get_freq(df, col):
    val = df
    val = val.groupby((col)).count()
    val.sort_values(by=val.columns[0], inplace=True, ascending=False)
    res = pd.DataFrame()
    res[col] = val.index
    res['count'] = val[val.columns[0]].tolist()
    res['percent'] = res['count'] / res['count'].sum()
    return res


def get_mat_vals(df):
    y = df.groupby(COL_N_ERROR).count()[df.columns[0]]
    y = pd.DataFrame(y)
    y.columns = ['count']
    y['err'] = y.index
    y['pre'] = y['err'].apply(lambda x: x.strip().split('->')[0])
    y['post'] = y['err'].apply(lambda x: x.strip().split('->')[1])
    y['mat'] = y['count'] / y.groupby(['pre'])['count'].transform('sum')
    return y


def get_heat(df, count=False):
    mat = get_mat_vals(df)
    print_to_log('got mat val')
    values = 'count' if count else 'mat'
    heat = mat.pivot(index='pre', columns='post', values=values).fillna(0)
    return heat


def save_heat(heat, lang):
    fmt = '.0f' if heat.values.max() > 1 else '.2g'  # integers or floats
    ax = sns.heatmap(heat, annot=True, fmt=fmt, cmap='Blues', linewidths=0.3, cbar_kws={'shrink': 0.8}, square=False)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(" Syntactic relations English - "+lang+"\n")
    fig = plt.gcf()
    plt.ylabel('English')
    plt.xlabel(lang)
    fig.set_size_inches(22, 12, forward=False)
    # plt.tight_layout()
    plt.savefig("graphs/heat_"+ lang +".png".replace(' ', '_'), dpi=120)
    plt.close()


def create_save_err_profile(df, lang, count=False):
    heat = get_heat(df, count)
    if heat.size == 0:
        print('empty heatmap!!', lang)
        return
    save_heat(heat, lang)


def save_err_profile_by_nationality(df, languages):
    for lang in languages:
        create_save_err_profile(df[df[COL_LANG] == lang], lang)
        print_to_log('saved heat', lang)


def save_err_profile_minus_avg_by_nationality(df, languages):
    # get values for every language:
    first = languages[0]
    all_mats = get_mat_vals(df[df[COL_LANG] == first])
    for lang in languages[1:]:
        mat = get_mat_vals(df[df[COL_LANG] == lang])
        all_mats = all_mats.join(mat.drop(['pre', 'post', 'count', 'err'], axis=1), how='left', rsuffix="_"+lang)
    # avg of all languages:
    avg_mat = all_mats.drop(['pre', 'post', 'count', 'err'], axis=1).fillna(0).mean(axis=1).reset_index()
    avg_mat.columns = ['err', 'mat']
    avg_mat['pre'] = avg_mat['err'].apply(lambda x: x.strip().split('->')[0])
    avg_mat['post'] = avg_mat['err'].apply(lambda x: x.strip().split('->')[1])
    heat = avg_mat.pivot(index='pre', columns='post', values='mat').fillna(0)
    save_heat(heat, ' avg')

    # every language minus the average:
    for lang in languages:
        mat = get_mat_vals(df[df[COL_LANG] == lang])
        # mat['mat'] = mat['mat'] - avg_mat['mat']
        mat['mat'] = mat.subtract(avg_mat.drop(['pre', 'post'], axis=1).set_index('err'))['mat'].dropna(axis=0).fillna(0)
        heat = mat.pivot(index='pre', columns='post', values='mat').fillna(0)
        save_heat(heat, lang+" (minus avg)")


def save_err_profile_by_nationality_level(df, languages, levels=None):
    for lang in languages:
        for level in levels:
            create_save_err_profile(df[(df[COL_LANG] == lang) & (df[COL_LEVEL] == level)], f'{lang}_{level}', False)


def per_level_bar(df, levels):
    """
    Creates a matrix where in each cell is a bar plot comparing every level for that score.
    :param df:
    :param levels: Which levels to include in graph (list of numbers. ) order is important
    :return:
    """
    # TODO: create multi graph plt, in each a bar plot of all levels for that error. manually.
    levels.sort()
    first = levels[0]
    all_mats = get_mat_vals(df[df[COL_LEVEL] == first])
    # all_mats.columns = [*all_mats.columns[:-1], all_mats.columns[-1] + '_' + first]
    for level in levels[1:]:
        mat = get_mat_vals(df[df[COL_LEVEL] == level])
        all_mats = all_mats.join(mat.drop(['pre', 'post', 'count', 'err'], axis=1), how='left', rsuffix="_" + level)
    all_mats = all_mats.drop(['count'], axis=1).fillna(0)
    N = len( all_mats['pre'].unique())
    fig, axes = plt.subplots(ncols=N, nrows=N,figsize=(30, 20) )
    fig.suptitle('matrix by level\n Proficiency: left to right.',
                 fontsize=14, fontweight='bold')
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x=all_mats.columns[3:], height=all_mats.iloc[i][all_mats.columns[3:]])
        ax.set_ylim([0,1.0])
        # ax.set_xticks([], [])
        # ax.set_yticks([], [])
    for ax, col in zip(axes[0], all_mats['pre'].unique()):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], all_mats['pre'].unique()):
        ax.set_ylabel(row, rotation=90, fontsize=16, size='large')
    for i, ax in enumerate(axes.flatten()):
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    fig.set_size_inches(25, 16, forward=False)
    # plt.tight_layout()
    plt.savefig("graphs/by_level.png", dpi=80)


if __name__ == '__main__':
    print_to_log('starting')
    pkl = 'total/total.pkl'
    df = pickle.load(open(pkl, 'rb'))
    df=df[df[COL_N_ERROR]!='R:OTHER'] # found one 'other'
    print_to_log('loaded')
    df = df.drop(columns=['orig'])
    freq_errors = get_freq(df, COL_N_ERROR)
    print_to_log('got freq errors')
    freq_langs = get_freq(df, COL_LANG)
    print_to_log('got freq languages')
    freq_levels = get_freq(df, COL_LEVEL)
    # levels = freq_levels[COL_LEVEL].to_list()[:-2]
    levels = freq_levels[COL_LEVEL].to_list()[:6]
    freq_err = freq_errors[COL_N_ERROR].to_list()[:20]
    # print(freq_langs)
    # print(freq_errors)

    total_heat = get_heat(df)
    save_heat(total_heat, 'test tight')
    print_to_log('got total heat')
    N = 3  # Number of languages to use (N largest in dataset. discard the smaller ones.)
    languages = freq_langs[COL_LANG].to_list()[:N]
    save_err_profile_by_nationality(df, languages)
    save_err_profile_minus_avg_by_nationality(df, languages)

    per_level_bar(df, levels) # the real one..!
    dimitry_path = 'external_data/en-zh_pos_cm_percent.csv'
    lang = 'cn'
    r = compare_dimitry(df, 'external_data/en-ru_pos_cm_percent.csv', 'ru')
    f = compare_dimitry(df, 'external_data/en-fr_pos_cm_percent.csv', 'fr')
    j = compare_dimitry(df, 'external_data/en-jp_pos_cm_percent.csv', 'jp')
    c = compare_dimitry(df, 'external_data/en-zh_pos_cm_percent.csv', 'cn')
    # compare_dimitry(df, 'external_data/en-ko_pos_cm_percent.csv', 'ko')
    print(r,f,j,c)




