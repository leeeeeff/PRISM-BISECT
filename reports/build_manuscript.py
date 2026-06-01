#!/usr/bin/env python3
"""
Build unified English manuscript from individual draft files.
Removes: Korean text, TODO sections, draft headers, internal file paths.
Keeps: all scientific English content, tables, equations, Supplementary tables.
"""
import re
import os

BASE = '/home/welcome1/sw1686/DIFFUSE/reports'
OUTPUT = os.path.join(BASE, 'manuscript_full_english.md')

# ── helpers ──────────────────────────────────────────────────────────────────

def has_korean(s):
    return bool(re.search(r'[가-힣ᄀ-ᇿ㄰-㆏]', s))

def read_file(fname):
    with open(os.path.join(BASE, fname), encoding='utf-8') as f:
        return f.read()

def clean_inline(line):
    """Remove internal path/source references from a line."""
    line = re.sub(r'\s*`/home/[^`]+`', '', line)
    line = re.sub(r'\s*`/tmp/[^`]+`', '', line)
    line = re.sub(r'\s*`reports/[^`]+`', '', line)
    line = re.sub(r'\bSource:\s+\S+\.(json|py|txt)[^.]*\.?', '', line)
    line = re.sub(r'\boutput:\s+`[^`]+`\.?', '', line)
    line = re.sub(r'\bScript:\s+`[^`]+`[,;.]?', '', line)
    line = re.sub(r'The negative control analysis is provided in\s+`[^`]+`\.?', '', line)
    line = re.sub(r'\*\[TODO:[^\]]*\]\*', '', line)  # inline TODO markers
    return line.rstrip()

def is_draft_meta(line):
    """True if line is a draft header / metadata line to drop."""
    patterns = [
        r'^# (Abstract Draft|Introduction —|Methods —|Results —|Discussion —|Citation List)',
        r'^\*\*Draft \d{4}',
        r'^\*\*2026-0[45]-\d{2}',
        r'^\*\*Word limit:',
        r'^\*\*Target:',
        r'^\*\*생성:',  # Korean — caught by has_korean too
    ]
    return any(re.match(p, line) for p in patterns)

# ── section processors ───────────────────────────────────────────────────────

def process_abstract(text):
    """Return only the Version 4 body (2026-05-20 version)."""
    lines = text.split('\n')
    in_v4 = False
    result = []
    for line in lines:
        if re.match(r'^## Version 4', line):
            in_v4 = True
            continue
        if in_v4:
            if re.match(r'^## (Version|Key numbers|Key fixes|TODO)', line):
                break
            if has_korean(line) or is_draft_meta(line):
                continue
            result.append(clean_inline(line))
    while result and result[-1].strip() in ('', '---'):
        result.pop()
    return '\n'.join(result)


def process_generic(text, hard_stop_pats=None, soft_skip_pats=None):
    """
    Generic section processor.
    hard_stop_pats : regex list — stop consuming lines entirely.
    soft_skip_pats : regex list — skip lines until the next section header.
    """
    hard_stop_pats = hard_stop_pats or []
    soft_skip_pats = soft_skip_pats or []

    default_soft = [
        r'^## Key numbers',
        r'^## Key fixes',
    ]
    soft_skip_pats = default_soft + soft_skip_pats

    lines = text.split('\n')
    result = []
    soft_skip = False
    hard_stop = False

    for line in lines:
        # Hard stop check
        for pat in hard_stop_pats:
            if re.match(pat, line):
                hard_stop = True
                break
        if hard_stop:
            break

        # Soft-skip start check
        for pat in soft_skip_pats:
            if re.match(pat, line):
                soft_skip = True
                break

        if soft_skip:
            # Exit on any real section header that is NOT a skip trigger
            if re.match(r'^#{1,3} ', line) and not any(re.match(p, line) for p in soft_skip_pats):
                soft_skip = False
            else:
                continue

        # Drop Korean lines and draft metadata
        if has_korean(line) or is_draft_meta(line):
            continue

        # Drop ⚠️ warning lines (annotation-only, no scientific content)
        if '⚠️' in line:
            continue

        result.append(clean_inline(line))

    while result and result[-1].strip() in ('', '---'):
        result.pop()
    return '\n'.join(result)


def process_references(text):
    """Extract clean reference entries from citation list."""
    lines = text.split('\n')
    result = []
    skip = False

    skip_section_pats = [
        r'^## Citations Still Needed',
        r'^## Format Notes',
        r'^## NIPSNAP1',
    ]

    for line in lines:
        # Section-level skip triggers
        for pat in skip_section_pats:
            if re.match(pat, line):
                skip = True
                break

        if skip:
            # Never re-enter after these terminal sections
            continue

        if has_korean(line) or is_draft_meta(line):
            continue
        if '⚠️' in line:
            continue
        # Drop checkmark / status lines
        if re.match(r'\s*[-•]\s*\*\*(수정|주의|Better:)', line):
            continue

        result.append(clean_inline(line))

    while result and result[-1].strip() in ('', '---'):
        result.pop()
    return '\n'.join(result)

# ── assemble ─────────────────────────────────────────────────────────────────

parts = []

parts.append(
    '# DIFFUSE: Deep Isoform Function Prediction Using Sequence Embeddings\n\n'
    '*Draft manuscript — compiled 2026-05-21*\n\n---'
)

# Abstract (Version 4 only)
parts.append('## Abstract\n\n' + process_abstract(read_file('abstract_draft_20260516.md')))

# Introduction
intro = process_generic(
    read_file('introduction_draft_20260516.md'),
    hard_stop_pats=[r'^## TODO'],
)
parts.append(intro)

# Methods (keep Data availability + Ethics; stop at TODO list)
methods = process_generic(
    read_file('methods_draft_20260516.md'),
    hard_stop_pats=[r'^## TODO before submission'],
)
parts.append(methods)

# Results (skip Key-numbers block mid-section; stop at TODO)
results = process_generic(
    read_file('results_draft_20260516.md'),
    hard_stop_pats=[r'^## TODO before submission'],
)
parts.append(results)

# Discussion
discussion = process_generic(
    read_file('discussion_draft_20260516.md'),
    hard_stop_pats=[r'^## TODO'],
)
parts.append(discussion)

# References
refs = process_references(read_file('citation_list_20260517.md'))
parts.append('## References\n\n' + refs)

# ── write ────────────────────────────────────────────────────────────────────

output = '\n\n---\n\n'.join(parts)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(output)

lines_n = output.count('\n')
size_kb = len(output.encode('utf-8')) / 1024
print(f"Written: {OUTPUT}")
print(f"Lines  : {lines_n:,}")
print(f"Size   : {size_kb:.1f} KB")
