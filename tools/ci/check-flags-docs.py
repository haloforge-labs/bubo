#!/usr/bin/env python3
"""Enforce STE-style rules for the generated release switch document only."""
import argparse
import re
from pathlib import Path

def check_flags_docs(path):
    errors = []
    forbidden = re.compile(r'\b(I|we|us|our|ours|my|mine|unfortunately|sadly|mistake|bug|TODO)\b', re.I)
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if forbidden.search(line):
            errors.append(f'{path}:{number}: forbidden wording')
        if line.startswith('|'):
            cells = re.split(r'(?<!\\)\|', line)[1:-1]
            if any(not cell.strip() for cell in cells):
                errors.append(f'{path}:{number}: empty table cell')
        else:
            cells = [line]
        for cell in cells:
            # Ignore punctuation inside inline code and decimal numbers.
            cell = re.sub(r'`[^`]*`', 'CODE', cell)
            for sentence in re.split(r'(?<!\d)[.!?](?:\s|$)', cell):
                if len(sentence.split()) > 20:
                    errors.append(f'{path}:{number}: sentence exceeds 20 words')
    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = check_flags_docs(args.root / 'docs/FLAGS.md')
    for error in errors: print(error)
    if not errors: print('PASS: FLAGS.md STE-style lint')
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
