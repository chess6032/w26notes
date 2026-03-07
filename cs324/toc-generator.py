"""
Table-of-contents generator for man-notes.md
"""

import re

MAN_NOTES_FILEPATH = './man-notes.md'
MAN_NOTES_FIRST_HEADER = "# man notes"
IS_FIRST_HEADER = lambda s: s.strip() == MAN_NOTES_FIRST_HEADER

def identify_headers(file) -> list:
    headers = []

    in_code_block = False
    for poopy_line in file:
        line = poopy_line.strip()
        
        if re.match("```[a-zA-Z0-9]*", line):
            in_code_block = not in_code_block
        if in_code_block:
            continue

        if re.match("#+[\\x20-\\xFF]+", line):
            headers.append(line)

    return headers

def strip_hashtags(header:str) -> tuple[int, str]:
    n = 0
    header = header.strip() # just in case
    while header[0] == '#':
        header = header[1:]
        n += 1
    return (n, header)

class Header:
    """
    A class representing a header that links to its subheaders.

    ### Attributes

    1. **name** (`str`): all text after the #s.
    2. **subheaders** (`list<Header>`): list of subheader Header objects.
    3. **header_num** (`int`): number of header if it were converted to an HTML tag. i.e., number of nestings. e.g., '###' is a "header 3", because it would be converted to \<h3>.
    """

    def __init__(self, name:str, header_num:int, subheaders:list=[]):

        self.name = name

        assert type(subheaders) == list
        assert type(subheaders[0]) == Header if subheaders else True

        self.subheaders = subheaders
        self.header_num = header_num

    def add_subheader(self, name:str): # -> Header
        sub = Header(name, self.header_num+1, [])
        self.subheaders.append(sub)
        return sub

def _nestify_headers(headers:list, superh:Header, nested_headers:list):
    # BASE CASE: previous Header object had no subheaders
    if not superh:
        return

    n, name = strip_hashtags(headers[1])
    if n == superh.header_num:

    _nestify_headers(, headers[1:])


def nestify_headers(headers:list) -> Header:
    """
    ### Args
    
    - headers (`list`): the lines of headers as they appear in the file, with # chars included.
        - e.g. ['# skibid', '## fee', '## foo', '### bar', '## fum', '# gyatt']

    ### Return


    - `Header`: recursive list where each element is (header, [subheaders]), with # chars and leading/trailing space(s) stripped.
        - e.g. [ ('skibidi', [ ('fee', []), ('foo', [ ('bar', []) ]), ('fum', []) ]), ('gyatt', []) ] ]
    """

    root = Header(None, 0, [])
    _nestify_headers(headers, root, [root])
    return root


def main() -> None:
    headers = []

    with open(MAN_NOTES_FILEPATH) as f:
        # move to `# man notes`
        while (not IS_FIRST_HEADER(next(f))):
            pass
        headers = identify_headers(f)
    
    print(headers)


if __name__ == "__main__":
    main()