import pandas as pd
from os.path import splitext
import pickle
from xml_parsing import xml_to_prl, prl_pickle




def pipeline(xml_path):
    """
    processes xml file, PRL file with metadata,
    then parses to a DataFrame and pickles it.
    :param xml_path: path to xml file
    :return: path to pickled DF.
    """
    XML_FILE_PATH = xml_path
    PRL_FILE_PATH = f'{splitext(XML_FILE_PATH)[0]}.prl'
    PKL_FILE_PATH = f'{splitext(XML_FILE_PATH)[0]}.pkl'

    # pipeline:
    # xml to parallel with meta
    xml_to_prl(XML_FILE_PATH, PRL_FILE_PATH, True)
    # pickle DF
    return prl_pickle(PRL_FILE_PATH, PKL_FILE_PATH)



if __name__ == '__main__':
    XML_FILE_PATH = "short_xml.xml"
    pkl = pipeline('short_xml.xml')
    pkl = pickle.load(open(pkl, 'rb'))
    pkl = pickle.load(open('short_xml.pkl', 'rb'))
    pkl = pkl.drop(columns=['orig'])
    print(pkl.head())