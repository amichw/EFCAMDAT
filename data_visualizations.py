import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score as mmi
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import v_measure_score as vms
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, pairwise_distances
from scipy.stats import pearsonr as prs, spearmanr
import lang2vec.lang2vec as l2v


COL_N_ERROR = 'new_error_types'
COL_LANG = 'learner_nationality'
COL_LEVEL = 'level'

country_to_lang = {'br':'pt', 'cn':'zh', 'mx':'es', 'ru':'ru', 'de':'de', 'it':'it', 'fr':'fr', 'jp':'ja', 'tr':'tr'}


def rank_correlation(errors_dist, lang2vec_dist, langs):
    corrs, pvs, lang_labels = [], [], []
    for lang in langs:
        if lang in country_to_lang:
            corr, pv = spearmanr(errors_dist[lang], lang2vec_dist[lang])
            corrs.append(corr)
            pvs.append(pv)
            lang_labels.append(lang)
    plt.bar(lang_labels, corrs)
    plt.title('spearman rank correlation of distances of language from other '
              'languages \n by error-profile vs. lang2vec profile')
    pv_str = [f'p value=\n{pv}' for pv in pvs]
    for i in range(len(corrs)):
        plt.annotate(pv_str[i], (i, 0.2), ha='center')
    plt.gcf().set_size_inches(25, 16, forward=False)
    plt.savefig("graphs/rank_correlation.png", dpi=80)



def kl(a, b):
    """
    Computes KL-divergence
    :param a:
    :param b:
    :return:
    """
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    a = 1.0 * a / np.sum(a, axis=0) # make prob. dist.
    b = 1.0 * b / np.sum(b, axis=0)

    return np.sum(np.where(np.logical_and(a != 0, b != 0), a * np.log(a / b), 0))


def mmi_score(l1, l2):
    #  1)get heat map 2. mmi for every line. 3. avg all lines for final mmi.
    mmi_per_row = []
    for i in range(l1.shape[0]):
        mmi_per_row.append(mmi(l1.iloc[i], l2.iloc[i]))
    return np.average(mmi_per_row)


def cosine_sim_per_row(l1, l2):
    #  1)get heat map 2. mmi for every line. 3. avg all lines for final mmi.
    cs_per_row = []
    for i in range(l1.shape[0]):
        cs_per_row.append(cosine_similarity(np.reshape(l1.iloc[i].array, (1, -1)), np.reshape(l2.iloc[i].array, (1, -1)))[0][0])
    return np.average(cs_per_row)


def distFromLang(all_heats, langs, target_lang):
    """
    measures the distance of all langs from a specific lang
    :return: DF of distances
    """
    distances = dict()
    distDF = pd.DataFrame()
    dfLangs = []
    dfDist = []
    # y.columns = ['count']
    target = all_heats[target_lang]
    for lang in langs:
        if lang in country_to_lang:
            mat = all_heats[lang]
            dfLangs.append(lang)
            dfDist.append(cosine_sim_per_row(mat, target))
            distances[f'{lang}_{target_lang}'] = cosine_sim_per_row(mat, target)
    distDF['from'] = dfLangs
    distDF[target_lang] = dfDist
    return distDF


def allDistFromLang(all_heats, langs):
    """
    measures the distance of all langs from all langs.
        Saves result.
    """
    allDist = None
    for lang in langs:
        if lang in country_to_lang:
            if allDist is None:
                allDist = distFromLang(all_heats, langs, lang)
            else:
                allDist[lang] = distFromLang(all_heats, langs, lang)[lang]
    allDist = allDist.set_index('from')
    title = 'Distances between languages, by error profiles. \n computed by cosine similarity.'
    saveHeatMap(allDist, title, fileName="graphs/distances_prs.png")
    return allDist


def saveHeatMap(mat, title='', x='', y='', fileName='temp.png'):
    fmt = '.4g'  # floats
    ax = sns.heatmap(mat, annot=True, fmt=fmt, cmap='Blues', linewidths=0.3,
                     cbar_kws={'shrink': 0.8}, square=False)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    fig = plt.gcf()
    plt.ylabel(y)
    plt.xlabel(x)
    fig.set_size_inches(22, 12, forward=False)
    plt.savefig(fileName, dpi=80)
    plt.close()


def allDistLang2vec(langs):
    """
    cosine similarity matrix for lang2vec.
    measures the distance of all langs from all langs, from Lang2Vec vector.
        Saves result.
    """
    l2vs = dict()
    distL2vCS = pd.DataFrame()
    dfLangs = []
    for lang in langs:
        if lang in country_to_lang:
            code = country_to_lang[lang]
            l2vs[lang] = l2v.get_features(code, 'learned')[code]
    distL2vCS['from'] = l2vs.keys()
    for lang1 in l2vs:
        dfDist = []
        for lang2 in l2vs:
            dfLangs.append(lang2)
            dfDist.append(cosine_similarity(np.reshape(l2vs[lang1], (1, -1)), np.reshape(l2vs[lang2], (1, -1)))[0][0])
        distL2vCS[lang1] = dfDist
    distL2vCS = distL2vCS.set_index('from')
    title = 'Distances between languages, by Lang2Vec. \n computed by cosine similarity.'
    saveHeatMap(distL2vCS, title, fileName="graphs/distances_prs_l2v.png")
    return distL2vCS


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
    plt.savefig("graphs/heat_" + lang + ".png".replace(' ', '_'), dpi=80)
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
    levels = freq_levels[COL_LEVEL].to_list()[:13]
    langs = freq_langs[COL_LANG].to_list()[:10]
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

    all_heats = dict()
    for lang in langs:
        if lang in country_to_lang:
            all_heats[lang] = get_heat(df[df[COL_LANG] == lang])

    errors_dist = allDistFromLang(all_heats, langs)
    lang2vec_dist = allDistLang2vec(langs)  # objective no. 1 . complete!
    # TODO: rank spearman (objective 2). one number? what is the output??
    rank_correlation(errors_dist, lang2vec_dist, langs)


