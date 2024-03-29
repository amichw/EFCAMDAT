from bs4.element import Tag
from bs4 import BeautifulSoup as bs
from os.path import splitext
import pickle
import errant
import pandas as pd
from time import time
import datetime
QUOT_ENCODING = '&quot;'

LOG_PATH = 'pipeline_log'


def print_to_log(*text, new_path=None):
    if new_path:
        global LOG_PATH
        LOG_PATH = new_path
    with open(LOG_PATH, 'a') as f:
        for t in text:
            f.write("".join(str(datetime.datetime.now()).split('.')[:-1])+' - ')
            f.write(str(t))
            f.write('\n')
    print(text)


def find_all(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def xml_to_prl(xml_path, out_path=None, metadata=False):
    if not out_path:
        out_path = f'{splitext(xml_path)[0]}.prl'
    print_to_log('Parsing XML:', new_path=f'{splitext(out_path)[0]}.log')

    with open(xml_path, "r") as f_read:
        content = f_read.readlines()

    with open(out_path, 'w') as f_write:
        content = "".join(content)
        soup = bs(content, "lxml")
        writings = soup.find_all("writing")
        total = len(writings)
        current_per = 0
        current_index = 0
        for writing in writings:
            current_index += 1
            if (current_index / total) * 100 > current_per:
                print_to_log(f"Finished {current_index - 1} out of {total}. {current_per}%")
                current_per += 5
            if metadata:
                id = writing['id']
                level = writing['level']
                unit = writing['unit']
                learner = writing.find('learner')
                learner_id = learner['id']
                learner_nationality = learner['nationality']
                grade = writing.find('grade').contents[0]
                date = writing.find('date').contents[0]
                topic_id = writing.find('topic')['id']
                f_write.write("M|||{}|||{}|||{}|||{}|||{}|||{}|||{}|||{}\n".format(id, level, unit, learner_id,
                                                                                   learner_nationality, grade,
                                                                                   topic_id, date))
            texts = writing.find("text").contents
            orig = ""
            cor = ""
            for text in texts:
                if isinstance(text, Tag):
                    is_correction = text.name == 'change'
                    if is_correction:
                        orig += ' ' +  text.find("selection").text.strip().replace(QUOT_ENCODING, '\'')
                        cor += ' ' +  text.find("correct").text.strip().replace(QUOT_ENCODING, '\'')
                else:
                    text = text.strip()
                    text = text.replace(QUOT_ENCODING, '\'')
                    indexes = find_all(text, '.')
                    # index = text.find('.')
                    if len(indexes) == 0:
                        orig += ' ' + text
                        cor += ' ' + text
                    else:  # end of sentence in the middle of text:
                        first = 0
                        for second in indexes:  # for multiple '.'
                            f_write.write(f"O {orig} {text[first:second + 1]} \n")
                            f_write.write(f"C {cor} {text[first:second + 1]} \n\n")
                            first = second + 1
                        orig, cor = text[second + 1:], text[second + 1:]  # for text after '.'
            if len(orig.strip()) > 0 and len(cor.strip()) > 0 :
                # case where text didn't end with '.'
                f_write.write(f"O {orig} \n")
                f_write.write(f"C {cor} \n\n")
    return out_path


def prl_to_pickle_and_m2(prl_path, pkl_path=None, error_indices=None, corpus_native=False):
    if not pkl_path:
        pkl_path = f'{splitext(prl_path)[0]}.pkl'
    temp_pkl = pkl_path + 'temp'
    m2_path = f'{splitext(prl_path)[0]}.m2'
    columns = ['text_index', 'orig', 'cor_type', 'edit.o_start', 'edit.o_end', 'o_str', 'c_str', 'id', 'level', 'unit', 'learner_id',
               'learner_nationality', 'grade',
               'topic_id', 'date']
    if corpus_native:
        columns = ['text_index', 'orig', 'cor_type', 'edit.o_start', 'edit.o_end', 'o_str', 'c_str', 'd', 'f', 'g', 'l', 'a', 'sent_id']
    data = pd.DataFrame(columns=columns)
    pickle.dump(data, open(temp_pkl, 'wb'))
    index = 0
    total = 0
    current_index = 0
    current_per = 0
    my_index = -1
    with open(prl_path) as f:
        for _ in f:
            total += 1
    print_to_log('total lines:', total)
    annotator = errant.load('en')
    start_time, last_time = time(), time()
    with open(m2_path, 'w') as f_m2:
        with open(prl_path, 'r') as f:
            print_to_log('Starting to read to DF: ')
            meta, original, cor = "", "", ""
            for line in f:
                current_index += 1  # just for displaying count
                if (current_index) * 100 >= current_per * total:
                    elapsed = int(time() - start_time)
                    remaining = ((100 -current_per)//5 )* int(time() - last_time)
                    d = remaining // (24 * 3600)
                    h = (remaining  % (24 * 3600)) // 3600
                    m = remaining % 3600 // 60
                    s = remaining % 60
                    print_to_log(f"Finished {current_index - 1} out of {total}. {current_per}%." + f' elapsed: {elapsed}'+
                            f' remaining: {remaining} ({d}d {h}:{m}:{s} )'+ f' last time: {int(time() - last_time)}')
                    current_per += 5
                    last_time = time()
                    # save and erase current df. make program run faster:
                    df = pickle.load(open(temp_pkl, 'rb'))
                    df = pd.concat([df, data])
                    pickle.dump(df, open(temp_pkl, 'wb'))
                    data = pd.DataFrame(columns=columns)
                    df = None
                    index = 0

                if len(line) <= 0:
                    continue
                if line[0] == 'M':
                    meta = line.split('|||')[1:]
                elif line[0] == 'O':
                    original = line[1:]
                    my_index += 1
                elif line[0] == 'C':
                    cor = line[1:]
                    orig = annotator.parse(original)
                    cor = annotator.parse(cor)
                    edits = annotator.annotate(orig, cor)
                    f_m2.write(f'\nS {orig} \n')
                    for edit in edits:
                        try:
                            row = [my_index, original, edit.type, edit.o_start, edit.o_end, edit.o_str, edit.c_str, *meta]
                            data.loc[index] = row
                            index += 1
                            f_m2.write(f'{edit.to_m2()} \n')
                        except:
                            print_to_log('exception!!!! ========= ')
    df = pickle.load(open(temp_pkl, 'rb'))
    df = pd.concat([df, data])
    df.reset_index(drop=True, inplace=True)
    pickle.dump(df, open(pkl_path, 'wb'))
    return pkl_path, m2_path, df


def prl_to_corpus(prl_path):
    out_path_orig = f'{splitext(prl_path)[0]}_orig'
    out_path_corr = f'{splitext(prl_path)[0]}_corr'

    with open(prl_path, "r") as f_read:
        with open(out_path_orig, 'w') as f_orig:
            with open(out_path_corr, 'w') as f_corr:
                for line in f_read:
                    # line = line.strip()
                    if len(line.strip()) <= 0:
                        continue
                    elif line[0] == 'O' and len(line[1:].strip()) > 0:
                        f_orig.write(line[1:])
                    elif line[0] == 'C':
                        f_corr.write(line[1:])
    return out_path_orig, out_path_corr


def get_errors(m2_path):
    errors = []
    with open(m2_path, 'r') as m2:
        for line in m2:
            line = line.strip()
            if len(line) <= 0: continue
            if line[0] == 'A':
                errors.append(line.split('|||')[1])
    return errors


def corpus_native_to_prl(cPath, out_path=None, metadata=False):
    if not out_path:
        out_path = f'{splitext(cPath)[0]}_c.prl'
    print_to_log('Parsing corpus:', new_path=f'{splitext(out_path)[0]}.log')

    with open(cPath, "r", encoding='utf-8') as f_read:
        content = f_read.readlines()

    with open(out_path, 'w') as f_write:
        # content = "".join(content)
        current_index = 0
        meta = {'d': '', 'f': '', 'g': '', 'l': '', 'a': '', 'sent_id': ''}

        for line in content:
            if len(line) <= 0:
                continue
            if line[0] == '#':
                meta[line[2]] =  line[6:].strip() if len(line) > 6 else ''
            elif line[0] == 'O':
                orig = line[2:].strip()
            elif line[0] == 'C':
                cor = line[2:].strip()
                current_index += 1
                if metadata:
                    f_write.write("M|||{}|||{}|||{}|||{}|||{}|||{}\n".format(meta['d'], meta['f'], meta['g'], meta['l'],
                                                                             meta['a'], meta['sent_id']))
                f_write.write(f"O  {orig} \n")
                f_write.write(f"C  {cor} \n\n")
    print(f"wrote {current_index} sentences to prl {out_path}.")
    return out_path
