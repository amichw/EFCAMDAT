# -*- coding: utf-8 -*-
from six import string_types
import argparse
from time import time
import numpy as np
# from ud_labels import LABEL_TO_ID, ID_TO_LABEL
from collections import namedtuple
# from utils import DataUtil, AttrDict
from ufal.udpipe import Model, Pipeline, ProcessingError
# import yaml
import sys
import codecs
import locale
import gzip
import os

IN = 0
OUT = 1
LOOP = 2
models = {}

CompactConlluWord = namedtuple(
    'CompactConlluWordO', ['index', 'head', 'label'])


# class ConlluWordObject():
#
#     def __init__(self, word_conllu_list):
#         (self.index, self.form, self.lemma, _, _, _, self.head,
#          self.label, _, _) = word_conllu_list.split("\t")
#         self.index = float(self.index) - 1
#         self.head = float(self.head) - 1
#
#     def compact_form(self):
#         return CompactConlluWord(self.index, self.head, self.label)


def load_model(model_name):
    if not model_name in models:
        models[model_name] = Model.load(model_name)
    return models[model_name]


def udpipe(input, model_name, outputfile=None, batch_size=256, verbose=False):
    """
    Parse text to Universal Dependencies using UDPipe.
    :param sentences: iterable of iterables of strings (one string per line)
    :param model_name: filename containing UDPipe model to load
    :param verbose: print extra information
    :return: iterable of lines containing parsed output
    """
    # In Python2, wrap sys.stdin and sys.stdout to work with unicode.
    if sys.version_info[0] < 3:

        encoding = locale.getpreferredencoding()
        sys.stdin = codecs.getreader(encoding)(sys.stdin)
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)

    if isinstance(input, string_types):
        f = open(input, "r")
        lines = f.readlines()
        f.close()
    else:
        lines = [line + "\n" for line in input]

    output = None
    if outputfile is not None:
        if not os.path.isdir(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile))
        output = open(outputfile, "w")
        encoding = locale.getpreferredencoding()
        # encoding = 'UTF-8'
        print("encoding", encoding, type(encoding))
        # output = codecs.getwriter(encoding)(output)

    model = load_model(model_name)
    if not model:
        raise ValueError("Invalid model: '%s'" % model_name)

    pipeline = Pipeline(model, "horizontal",
                        Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    batches = [lines[x:x + batch_size]
               for x in range(0, len(lines), batch_size)]
    results = []
    for i, batch in enumerate(batches):
        text = "".join(batch)
        error = ProcessingError()
        num_tokens = sum(1 for l in batch if l)
        print("Running UDPipe on %d tokens, batch %d " %
              (num_tokens, i))
        start = time()
        processed = pipeline.process(text, error)
        duration = time() - start
        print("Done (%.3fs, %.0f tokens/s)" %
              (duration, num_tokens / duration if duration else 0))

        if verbose:
            print(processed)
        if error.occurred():
            raise RuntimeError(error.message)

        if output:
            to_write = "\n".join(processed.splitlines())
            if sys.version_info[0] >= 3:
                try:
                    to_write = bytes(to_write, encoding, errors='ignore')
                    to_write = to_write.decode(encoding, errors='ignore')
                except AttributeError:
                    print("to_write:",type(to_write), to_write[:100])
                to_write = str(to_write)
            # to_write = str(to_write)
            # output.write(to_write)
            # if verbose:
            #     print('DONE!', i)
            if(to_write and isinstance(to_write, str) and len(to_write)>0):
                output.write(to_write)
                if verbose:
                    print('DONE!', i)

        results.extend(processed.splitlines())

    if output:
        output.close()

    sys.stdout = sys.__stdout__
    sys.stdin = sys.__stdin__

    return results


# def split_conllu_to_sentences(conllu_list):
#     sentences = []
#     new_sentence = []
#     i = 0
#     while i < len(conllu_list):
#
#         while i < len(conllu_list) and not conllu_list[i].startswith("# sent_id"):
#             if conllu_list[i].startswith("# ") or len(conllu_list[i]) == 0:
#                 i = i + 1
#             elif(len(conllu_list[i]) != 0):
#                 try:
#                     cw = ConlluWordObject(conllu_list[i])
#                     new_sentence.append(cw.compact_form())
#                 except:
#                     print(conllu_list[i], "was not parsed correctly, probably due to two tokens for one original word")
#                 i = i + 1
#         else:
#             i = i + 1
#             if len(new_sentence) > 0:
#                 sentences.append(new_sentence)
#                 new_sentence = []
#     return sentences
#
# def extract_text_from_conllu(conllu_list):
#     sentences = []
#     new_sentence = []
#     i = 0
#     while i < len(conllu_list):
#
#         while i < len(conllu_list) and not conllu_list[i].startswith("# sent_id"):
#             if conllu_list[i].startswith("# ") or len(conllu_list[i]) == 0:
#                 i = i + 1
#             elif(len(conllu_list[i]) != 0):
#                 try:
#                     new_sentence.append(conllu_list[i].split("\t")[1])
#                 except:
#                     print(conllu_list[i], "was not parsed correctly, probably due to two tokens for one original word")
#                 i = i + 1
#         else:
#             i = i + 1
#             if len(new_sentence) > 0:
#                 sentences.append(" ".join(new_sentence))
#                 new_sentence = []
#     return sentences

# def from_conllu_to_edges(conllu_sentence, max_words_num=None):
#     if max_words_num is None:
#         max_words_num = len(conllu_sentence)
#
#     weights = np.zeros((max_words_num, max_words_num, 3))
#     for word in conllu_sentence:
#         weights[word.index, word.head, IN] = 1
#         weights[word.head, word.index, OUT] = 1
#         weights[word.index, word.index, LOOP] = 1
#
#     return weights


# def from_conllu_to_bias(conllu_sentence, max_words_num=None):
#     if max_words_num is None:
#         max_words_num = len(conllu_sentence)
#
#     bias = np.zeros((max_words_num, max_words_num, len(LABEL_TO_ID.keys())))
#     for word in conllu_sentence:
#         bias[word.index, word.head, LABEL_TO_ID[word.label]] = 1
#
#     return bias

#
# def udpipe_parsing(input, model_name, conllu_output=None, batch_size=256, verbose=False):
#
#     conllu_list = udpipe(input, model_name, conllu_output, batch_size, verbose)
#     sentences = split_conllu_to_sentences(conllu_list)
#     return sentences


# def convert_parallel(pud_de_path="/cs/snapless/oabend/borgr/SSMT/data/UD/de_pud-ud-test.conllu"):
#
#     # pud_en_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/en_pud-ud-test.conllu"
#     en = []
#     de = []
#     with open(pud_de_path) as fl:
#         for line in fl:
#             de_pref = "# text = "
#             if line.startswith(de_pref):
#                 de.append(line[len(de_pref):])
#
#             en_pref = "# text_en = "
#             if line.startswith(en_pref):
#                 en.append(line[len(en_pref):])
#
#     with open(pud_de_path + ".txt", "w") as fl:
#         fl.writelines(de)
#
#     with open(pud_en_path + ".txt", "w") as fl:
#         fl.writelines(en)



def to_cp1255(path):
    with open(path,'r', encoding='UTF-8') as f:
        txt = f.readlines()
        txt = "".join(txt)
        # txt = strip_accents(txt)
        with open('converted_{}'.format(path), 'w', encoding='cp1255') as f_out:
            f_out.write(txt)

#
# if __name__ == '__main__':
#     print('starting')
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', dest='config')
#     parser.add_argument('-i', '--input', dest='input')
#     parser.add_argument('-m', '--model', dest='model')
#     parser.add_argument('-o', '--output', dest='output')
#     parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=10000)
#     parser.add_argument('-f', '--force', dest='force', action="store_true")
#     args = parser.parse_args()
#     if args.config:
#         pass
#         # config = AttrDict(yaml.load(open(args.config)))
#         # input_file, model_file, conllu_output, batch_size = config.input_file, config.model_file, config.conllu_output, config.batch_size
#         # args.force=True
#     else:
#         input_file = args.input
#         model_file = args.model
#         conllu_output = args.output
#         batch_size = args.batch_size
#     # to_cp1255(input_file)
#     print("parsing" + input_file)
#     if args.force or not os.path.isfile(conllu_output):
#         udpipe_parsing(input_file, model_file, conllu_output, batch_size, True)
#     else:
#         print("conllu file already exists, pass -f or --force to overwrite" + conllu_output)
#     # udpipe_parsing(["Parliament Does Not Support Amendment Freeing Tymoshenko", "its ratification would require 226 votes ."], '/cs/usr/zohara/PycharmProjects/SSMT/udpipe/english-ud-2.0-170801.udpipe', "test.conllu", 1000)
