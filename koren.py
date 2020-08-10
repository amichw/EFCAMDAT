import re
from bs4.element import Tag
from bs4 import BeautifulSoup as bs
from os.path import splitext

QUOT_ENCODING = "&quot;"


def convert_to_m2(xml_path, m2_path=None):
    if not m2_path:
        m2_path = f'{splitext(xml_path)[0]}.m2'

    with open(xml_path, "r") as f_read:
        with open(m2_path, 'w') as f_write:
            content = f_read.readlines()
            content = "".join(content)
            soup = bs(content, "lxml")
            writings = soup.find_all("writing")
            for writing in writings:
                texts = writing.find("text").contents
                sentences = writing.find_all("sentence")

                sent_errors_indices, sent_correct_phrases, sent_correct_symbols = list(), list(), list()
                cur_token_id, cur_sent_id = 0, 0
                tokens = sentences[cur_sent_id].find_all("token")

                for text in texts:
                    is_correction = isinstance(text, Tag)

                    if is_correction:
                        first_token_id = cur_token_id
                        incorrect_phrase_tag = text.find("selection")
                        try:
                            phrase = incorrect_phrase_tag.text.strip()
                        except:
                            continue
                    else:
                        phrase = text.strip() if isinstance(text, str) else None
                    while phrase:
                        if phrase.startswith(tokens[cur_token_id].text):
                            phrase = phrase[len(tokens[cur_token_id].text):].strip()
                            cur_token_id += 1

                        # Case of different encoding in the phrase vs the text
                        elif phrase.startswith(QUOT_ENCODING) and tokens[cur_token_id].text in ['``', '\'\'']:
                            phrase = phrase[len(QUOT_ENCODING):].strip()
                            cur_token_id += 1

                        # Case of ".<word>" token
                        elif tokens[cur_token_id].text.startswith("."):
                            phrase = phrase[len("."):].strip()
                            tokens[cur_token_id].string = tokens[cur_token_id].text[len("."):]

                        else:
                            # If tokenization started from the middle of the phrase
                            if tokens[cur_token_id].text in phrase and tokens[cur_token_id + 1].text in phrase:
                                new_index = re.search(rf'\b({tokens[cur_token_id].text})\b', phrase)
                                phrase = phrase[new_index.start():]

                            else:
                                # print(
                                #     f"In writing_id={writing.attrs['id']} sentence_id={sentences[cur_sent_id].attrs['id']}"
                                #     f" token_id={tokens[cur_token_id].attrs['id']}")
                                break  # Move to the next text

                        if cur_token_id >= len(tokens):

                            original_sentence = " ".join([token.text for token in tokens])
                            f_write.write(f"S {original_sentence}\n")
                            for j in range(len(sent_errors_indices)):
                                f_write.write(
                                    f"A {sent_errors_indices[j]}|||{sent_correct_symbols[j]}|||{sent_correct_phrases[j]}|||REQUIRED|||-NONE-|||0\n")
                            f_write.write('\n')

                            sent_correct_phrases, sent_correct_symbols, sent_errors_indices = list(), list(), list()

                            if cur_sent_id + 1 < len(sentences):
                                cur_sent_id += 1
                                tokens, cur_token_id = sentences[cur_sent_id].find_all("token"), 0

                    if is_correction:
                        sent_correct_symbols.append(text.find("symbol").text)
                        sent_correct_phrases.append(text.find("correct").text.strip())
                        sent_errors_indices.append((first_token_id, max(first_token_id, cur_token_id)))
