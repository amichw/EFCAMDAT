import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
            create_save_err_profile(df[(df[COL_LANG] == lang) & (df[COL_LEVEL] == level)], f'{lang}_{level}')


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
    # print(freq_langs)
    # print(freq_errors)

    total_heat = get_heat(df)
    save_heat(total_heat, 'test tight')
    print_to_log('got total heat')
    N = 3  # Number of languages to use (N largest in dataset. discard the smaller ones.)
    languages = freq_langs[COL_LANG].to_list()[:N]
    save_err_profile_by_nationality(df, languages)
    save_err_profile_minus_avg_by_nationality(df, languages)





