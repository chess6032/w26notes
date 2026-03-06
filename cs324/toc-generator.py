"""
Table-of-contents generator for man-notes.md
"""

import re

MAN_NOTES_FILEPATH = './man-notes.md'

def regex_search_file(filepath: str, pattern: str) -> list:
    txt = ''
    with open(filepath) as f:
        txt = f.read()
    return re.findall(pattern, txt)

if __name__ == "__main__":
    x = regex_search_file(MAN_NOTES_FILEPATH, "^##*[^\n]*") # gets close but matches w/ #include lines in code
                                                            # uhhh wait actually that's awkward it works on regex101.com but not here w/ in Python
    print(x)