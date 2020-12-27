import pandas as pd
from os.path import splitext, exists, join
from os import mkdir
import pickle
from sys import argv
from xml_parsing import xml_to_prl, prl_to_pickle_and_m2, prl_to_corpus, get_errors
from ufal_stuff.udpipe import udpipe
from ufal_stuff.GEC_UD_divergences_m2 import run_gec
import datetime
# from ufal_stuff.preprocessing import create_corrected_sentences
# from ufal_stuff.preprocessing import preprocess_file


def print_to_log(*text):
    with open('pipeline_log', 'a') as f:
        for t in text:
            f.write("".join(str(datetime.datetime.now()).split('.')[:-1])+' - ')
            f.write(str(t))
            f.write('\n')
    print(text)

def pipeline(xml_path):
    """
    processes xml file, PRL file with metadata,
    then parses to a DataFrame and pickles it.
    create m2 and corpus
     create 'connlu' files of corpus
     print command to run in terminal to get new m2 file and confusion matrix
    :param xml_path: path to xml file
    :return: path to pickled DF.
    """
    # plan:
    # XML to  meta.prl
    # meta.prl to df
    # meta.prl to 1: orig, 2:corr
    # meta.prl to m2
    # udpipe: orig ->orig.connlu , corr.connlu
    # UD(orig.connlu , corr.connlu, m2, model)-> new m2
    # add new m2 error_types to df.

    xml_name = xml_path.split('/')[-1].split('.')[0]
    dirName = "".join(xml_path.split('/')[:-1])
    new_dir = join(dirName, xml_name)
    if not exists(new_dir):
        mkdir(new_dir)
    # dirName = join(xml_path.split('/')[-1], dirName)
    prl_file_path = join(new_dir, f'{xml_name}.prl')
    pkl_file_path = join(new_dir, f'{xml_name}.pkl')
    # PKL_FILE_PATH = f'{dirName}.pkl'
    # m2_path = f'{splitext(XML_FILE_PATH)[0]}.m2'
    model = 'ufal_stuff/english-ewt-ud-2.5-191206.udpipe'

    # xml to parallel with meta
    xml_to_prl(xml_path, prl_file_path, True)
    # pickle DF, create m2 using errant:
    pkl, m2_path, df = prl_to_pickle_and_m2(prl_file_path, pkl_file_path)
    # meta.prl to 1: orig, 2:corr
    orig, corr = prl_to_corpus(prl_file_path)
    #  parallel =>  connluX2
    # udpipe: orig ->orig.connlu , corr.connlu
    connlu_orig_path = f'{orig}.connlu'
    connlu_corr_path = f'{corr}.connlu'
    print_to_log('creating connlu files:=============')
    res = udpipe(orig, model, connlu_orig_path, 256, False)
    res = udpipe(corr, model, connlu_corr_path, 256, False)
    # UD(orig.connlu , corr.connlu, m2, model)-> new m2 :
    print_to_log('now run in terminal: ', f'python ufal_stuff/GEC_UD_divergences_m2.py {connlu_orig_path} {connlu_corr_path} {m2_path}')
    print_to_log('running:', f'python ufal_stuff/GEC_UD_divergences_m2.py {connlu_orig_path} {connlu_corr_path} {m2_path} ')
    new_m2_path, invalid_indices = run_gec(connlu_orig_path, connlu_corr_path, m2_path)
    print_to_log('now running: add_new_errors() ')
    #  run add_new_error_types():
    df, pkl = add_new_error_types(df, m2_path, new_m2_path, pkl, invalid_indices)
    return pkl, df, m2_path


def add_new_error_types(df, m2_path, new_m2_path, pkl, invalid_indices=""):
    # exclude invalid
    for i in invalid_indices:
        df = df[df['text_index'] != i]
    # add new m2 error_types to df:
    # pkl = f'{m2_path}.pkl'
    # new_m2_path = splitext(m2_path)
    # new_m2_path = "".join([new_m2_path[0], ".stx", new_m2_path[1]])  # this is what GEC_UD creates
    df['new_error_types'] = get_errors(new_m2_path)
    pickle.dump(df, open(pkl, 'wb'))
    return df, pkl



if __name__ == '__main__':
    if (len(argv) != 2):
        print_to_log("Usage: <xml file>")
    else:
        xml_path = argv[1]
        pkl, df, m2 = pipeline(xml_path)
        df = pickle.load(open(pkl, 'rb'))
        df = df.drop(columns=['orig'])
        print_to_log(df.head())
    exit(42)

    XML_FILE_PATH = "short_xml.xml"
    pkl = pipeline('short_xml.xml')
    pkl = pickle.load(open(pkl, 'rb'))
    pkl = pickle.load(open('short_xml.pkl', 'rb'))
    pkl = pkl.drop(columns=['orig'])
    print(pkl.head())