with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Line 1269 (index 1268) currently ends with ',\n'
# It needs to end with '))\n' to close go.Scatterpolar() and add_trace()
print('Line 1269 before:', repr(lines[1268]))

lines[1268] = "                fillcolor=FILL_COLORS.get(nm, 'rgba(0,212,255,0.15)')))\n"

print('Line 1269 after :', repr(lines[1268]))

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('SAVED')

import py_compile
try:
    py_compile.compile('dashboard.py', doraise=True)
    print('SYNTAX OK')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
    with open('dashboard.py', encoding='utf-8') as f:
        ls = f.readlines()
    import re
    m = re.search(r'line (\d+)', str(e))
    if m:
        ln = int(m.group(1))
        for j,l in enumerate(ls[max(0,ln-4):ln+4],start=max(1,ln-3)):
            print(f'{j:4d}: {repr(l)}')
