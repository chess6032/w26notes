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