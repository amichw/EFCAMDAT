from bs4.element import Tag
from bs4 import BeautifulSoup as bs
from os.path import splitext
import pickle
import errant
import pandas as pd

QUOT_ENCODING = '&quot;'


def find_all(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def xml_to_prl(xml_path, out_path=None, metadata=False):
    print('Parsing XML:')
    if not out_path:
        out_path = f'{splitext(xml_path)[0]}.prl'

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
                print(f"Finished {current_index - 1} out of {total}. {current_per}%")
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
                # print(id, level, unit, learner_id, learner_nationality, grade, topic_id, date)
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
            if len(orig) > 0:  # case where text didn't end with '.'
                f_write.write(f"O {orig} \n")
                f_write.write(f"C {cor} \n\n")


def prl_pickle(prl_path, out_path=None):
    if not out_path:
        out_path = f'{splitext(prl_path)[0]}.pkl'
    data = pd.DataFrame(
        columns=['orig', 'cor_type', 'cor_s', 'id', 'level', 'unit', 'learner_id', 'learner_nationality', 'grade',
                 'topic_id', 'date'])
    index = 0
    total = 0
    current_index = 0
    current_per = 0
    with open(prl_path) as f:
        for _ in f:
            total += 1
    print('total lines:', total)
    annotator = errant.load('en')
    with open(prl_path, 'r') as f:
        print('Starting to read to DF: ')
        meta, original, cor = "", "", ""
        read_orig = False

        for line in f:
            current_index += 1  # just for displaying count
            if (current_index) * 100 >= current_per * total:
                print(f"Finished {current_index - 1} out of {total}. {current_per}%")
                current_per += 5

            if len(line) <= 0:
                continue
            if line[0] == 'M':
                meta = line.split('|||')[1:]
            # elif not read_orig:
            elif line[0] == 'O':
                original = line[1:]
                read_orig = True
            elif line[0] == 'C':
                cor = line[1:]
                read_orig = False
                orig = annotator.parse(original)
                cor = annotator.parse(cor)
                edits = annotator.annotate(orig, cor)
                for edit in edits:
                    try:
                        row = [original, edit.type, edit.c_str, *meta]
                        data.loc[index] = row
                        index += 1
                    except:
                        print('exception!!!! ========= ')
    pickle.dump(data, open(out_path, 'wb'))
    return out_path
