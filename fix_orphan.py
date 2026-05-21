with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 1270-1273 (index 1269-1272) which are orphan broken code
# Line 1269 = index 1268 = correct fillcolor line (keep)
# Lines 1270-1273 = index 1269-1272 = orphan lines (delete)

print('BEFORE fix - lines 1268-1274:')
for i, line in enumerate(lines[1267:1274], start=1268):
    print(f'{i:4d}: {repr(line)}')

# Delete lines at index 1269, 1270, 1271, 1272
del lines[1269:1273]

print('\nAFTER fix - lines 1268-1274:')
for i, line in enumerate(lines[1267:1274], start=1268):
    print(f'{i:4d}: {repr(line)}')

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('\nSAVED dashboard.py')

import py_compile
try:
    py_compile.compile('dashboard.py', doraise=True)
    print('SYNTAX OK - no errors')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
