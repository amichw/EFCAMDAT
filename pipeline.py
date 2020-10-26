import pandas as pd
from os.path import splitext
import pickle
from xml_parsing import xml_to_prl, prl_to_pickle_and_m2, prl_to_corpus
from ufal_stuff.udpipe import udpipe




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
    # TODO: add new m2 error_types to df.

    XML_FILE_PATH = xml_path
    PRL_FILE_PATH = f'{splitext(XML_FILE_PATH)[0]}.prl'
    PKL_FILE_PATH = f'{splitext(XML_FILE_PATH)[0]}.pkl'
    model = 'ufal_stuff/english-ewt-ud-2.5-191206.udpipe'

    # xml to parallel with meta
    xml_to_prl(XML_FILE_PATH, PRL_FILE_PATH, True)
    # pickle DF, create m2
    pkl, m2_path = prl_to_pickle_and_m2(PRL_FILE_PATH, PKL_FILE_PATH)
    # meta.prl to 1: orig, 2:corr
    orig, corr = prl_to_corpus(PRL_FILE_PATH)
    #  m2 =>  connluX2
    # udpipe: orig ->orig.connlu , corr.connlu
    connlu_orig_path = f'ufal_stuff/{orig}.connlu'
    connlu_corr_path = f'ufal_stuff/{corr}.connlu'
    udpipe(orig, model, connlu_orig_path, 256, True)
    udpipe(corr, model, connlu_corr_path, 256, True)
    # UD(orig.connlu , corr.connlu, m2, model)-> new m2 :
    print('now run in terminal: ', f'python ufal_stuff/GEC_UD_divergences_m2.py {connlu_orig_path} {connlu_corr_path} {m2_path}')
    return pkl




if __name__ == '__main__':
    XML_FILE_PATH = "short_xml.xml"
    pkl = pipeline('short_xml.xml')
    pkl = pickle.load(open(pkl, 'rb'))
    pkl = pickle.load(open('short_xml.pkl', 'rb'))
    pkl = pkl.drop(columns=['orig'])
    print(pkl.head())