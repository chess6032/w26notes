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
    return (n, header.strip())

class Header:
    """
    A class representing a header that links to its subheaders.

    ### Attributes

    1. **name** (`str`): all text after the #s.
    2. **level** (`int`): number of nestings. i.e., the number of header if it were converted to an HTML tag. e.g., '###' is a "header 3", because it would be converted to \<h3>.
    3. **first_child** (`Header` | `None` ): the first of this header's subheaders, if any.
    4. **next_sibling** (`Header` | `None` ): the next equal-level header, if any.
    """

    def __init__(self, name:str, level:int, first_child=None, next_sibling=None):

        self.name = name
        self.level = level
        self.first_child = first_child
        self.next_sibling = next_sibling

    def __str__(self):
        return f"{self.level} - {self.name}"

    def ind_str(self):
        return f"{' ' * (self.level-1) * 4}{self.name}"

    def bear_child(self, name): # -> Header
        """
        Instantiates header as first child.

        name (`str`): name of new header.
        """

        if (self.first_child):
            raise RuntimeError(f"{str(self)} already has a child: {str(self.first_child)}")

        self.first_child = Header(name, self.level+1)
        return self.first_child

    def create_sibling(self, name):
        """
        Instantiates header object as sibling.

        name (`str`): name of new header.
        """
        if (self.next_sibling):
            raise RuntimeError(f"{str(self)} already has a sibling: {str(self.next_sibling)}")

        self.next_sibling = Header(name, self.level)
        return self.next_sibling

def _nestify_headers(prev:Header, headers:list):
    curr = prev

    while True:
        if not headers:
            # no more heeaders
            return

        n, name = strip_hashtags(headers[0])

        if n < curr.level:
            # reached end of siblings
            return # BACKTRACK

        headers.pop(0)

        if n > curr.level:
            # create child
            child = curr.bear_child(name)
            _nestify_headers(child, headers)

        elif n == curr.level:
            curr.create_sibling(name)
            curr = curr.next_sibling



def nestify_headers(headers:list) -> Header:
    """
    ### Args
    
    - headers (`list`): the lines of headers as they appear in the file, with # chars included.
        - e.g. ['# skibid', '## fee', '## foo', '### bar', '## fum', '# gyatt']

    ### Return


    - `Header`: recursive list where each element is (header, [subheaders]), with # chars and leading/trailing space(s) stripped.
        - e.g. [ ('skibidi', [ ('fee', []), ('foo', [ ('bar', []) ]), ('fum', []) ]), ('gyatt', []) ] ]
    """

    root = Header(None, 1, [])
    _nestify_headers(root, headers)
    return root


def _print_Headers(header:Header) -> None:
    if not header:
        return

    print(header.ind_str())
    _print_Headers(header.first_child)
    _print_Headers(header.next_sibling)

def print_Headers(first_header:Header) -> None:
    _print_Headers(first_header)


def main() -> None:
    headers = []

    with open(MAN_NOTES_FILEPATH) as f:
        # move to `# man notes`
        while (not IS_FIRST_HEADER(next(f))):
            pass
        headers = identify_headers(f)
    
    root = nestify_headers(headers)
    print_Headers(root)


if __name__ == "__main__":
    main()