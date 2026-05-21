with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Replace lines 628-632 (index 627-631) directly
# Old: st.dataframe with background_gradient subset
# New: safe dataframe with no gradient subset issue

lines[627] = '    st.dataframe(\n'
lines[628] = '        df_m[dcols].style.format(fmt),\n'
lines[629] = '        use_container_width=True)\n'
lines[630] = '\n'
lines[631] = '\n'

print('AFTER fix lines 627-633:')
for i,l in enumerate(lines[626:634], start=627):
    print(f'{i:4d}: {repr(l[:90])}')

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('SAVED')

import py_compile
try:
    py_compile.compile('dashboard.py', doraise=True)
    print('SYNTAX OK')
except py_compile.PyCompileError as e:
    print(f'SYNTAX ERROR: {e}')
