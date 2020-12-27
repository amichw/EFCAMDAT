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
import datetime

IN = 0
OUT = 1
LOOP = 2
models = {}

CompactConlluWord = namedtuple(
    'CompactConlluWordO', ['index', 'head', 'label'])


def print_to_log(*text):
    with open('pipeline_log', 'a') as f:
        for t in text:
            f.write("".join(str(datetime.datetime.now()).split('.')[:-1])+' - ')
            f.write(str(t))
            f.write('\n')
    print(text)

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
    print_to_log(f'total batches: {len(batches)}')
    for i, batch in enumerate(batches):
        text = "".join(batch)
        error = ProcessingError()
        num_tokens = sum(1 for l in batch if l)
        print_to_log("Running UDPipe on %d tokens, batch %d " %
              (num_tokens, i))
        start = time()
        processed = pipeline.process(text, error)
        duration = time() - start
        print_to_log("Done (%.3fs, %.0f tokens/s)" %
              (duration, num_tokens / duration if duration else 0))

        if verbose:
            print_to_log(processed)
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
                    print_to_log('DONE!', i)

        results.extend(processed.splitlines())

    if output:
        output.close()

    sys.stdout = sys.__stdout__
    sys.stdin = sys.__stdin__

    return results


def to_cp1255(path):
    with open(path,'r', encoding='UTF-8') as f:
        txt = f.readlines()
        txt = "".join(txt)
        # txt = strip_accents(txt)
        with open('converted_{}'.format(path), 'w', encoding='cp1255') as f_out:
            f_out.write(txt)
