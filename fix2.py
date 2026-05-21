with open('dashboard.py', encoding='utf-8') as f:
    content = f.read()

fixes = [
    ("best_row['Model']",   "best_row.name"),
    ('best_row["Model"]',   "best_row.name"),
    ("row['Model']",        "row.name"),
    ('row["Model"]',        "row.name"),
]

count = 0
for old, new in fixes:
    if old in content:
        n = content.count(old)
        content = content.replace(old, new)
        print(f'  Replaced {n}x: {old!r} -> {new!r}')
        count += n

if count == 0:
    print('WARNING: No replacements made.')
    print('Searching for Model references near line 802...')
    lines = content.split('\n')
    for i, line in enumerate(lines[795:810], start=796):
        print(f'  {i:4d}: {repr(line)}')
else:
    with open('dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'\nSAVED: {count} fix(es) applied to dashboard.py')

# Verify
with open('dashboard.py', encoding='utf-8') as f:
    check = f.read()
if "best_row['Model']" in check or 'best_row["Model"]' in check:
    print('ERROR: Still contains best_row[Model]')
else:
    print('VERIFIED: No more best_row[Model] references')
