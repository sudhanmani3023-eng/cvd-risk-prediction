def _hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return f'rgba(0,212,255,{alpha})'

FILL_COLORS = {
    'Logistic Regression': _hex_to_rgba('#00d4ff', 0.15),
    'Random Forest':       _hex_to_rgba('#00ff88', 0.15),
    'XGBoost':             _hex_to_rgba('#ffd700', 0.15),
}

with open('dashboard.py', encoding='utf-8') as f:
    content = f.read()

# Add helper at top if missing
if '_hex_to_rgba' not in content:
    insert = '''
def _hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return f'rgba(0,212,255,{alpha})'

FILL_COLORS = {
    'Logistic Regression': _hex_to_rgba('#00d4ff', 0.15),
    'Random Forest':       _hex_to_rgba('#00ff88', 0.15),
    'XGBoost':             _hex_to_rgba('#ffd700', 0.15),
}
'''
    idx = content.find('\nst.set_page_config')
    content = content[:idx] + insert + content[idx:]
    print('ADDED: _hex_to_rgba and FILL_COLORS')

# Fix ALL fillcolor references - replace entire radar trace block
import re

# Replace any fillcolor= line that contains rgba(00 or rgba( without proper format
def fix_fillcolor(m):
    name_var = m.group(1).strip()
    return f"fillcolor=FILL_COLORS.get({name_var}, 'rgba(0,212,255,0.15)'"

# Simple targeted replacement of the bad fillcolor line
old_patterns = [
    "fillcolor=COLORS.get(nm,'rgba(0,212,255,0.1)').replace(\n                    '#','rgba(').rstrip(')')\n                + ',0.1)' if '#' in COLORS.get(\n                    nm,'') else 'rgba(0,212,255,0.1)')",
    "fillcolor=_hex_to_rgba(\n                    COLORS.get(nm,'#00d4ff'), 0.15))",
    "fillcolor=_hex_to_rgba(COLORS.get(nm,'#00d4ff'), 0.15))",
]
new_fc = "fillcolor=FILL_COLORS.get(nm, 'rgba(0,212,255,0.15)')"

replaced = False
for old in old_patterns:
    if old in content:
        content = content.replace(old, new_fc)
        print(f'REPLACED: fillcolor pattern')
        replaced = True
        break

if not replaced:
    # Find and replace by line number approach
    lines = content.split('\n')
    new_lines = []
    skip_next = 0
    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue
        if 'fillcolor' in line and ('rgba(00' in line or
            ("COLORS.get" in line and 'fillcolor' in line)):
            # Replace this line and potentially next few
            new_lines.append(
                "                fillcolor=FILL_COLORS.get("
                "nm, 'rgba(0,212,255,0.15)'),")
            # Check if next lines are continuation
            j = i + 1
            while j < len(lines) and j < i + 5:
                if lines[j].strip().startswith('+') or \
                   lines[j].strip().startswith("'rgba") or \
                   ('else' in lines[j] and 'rgba' in lines[j]):
                    skip_next += 1
                    j += 1
                else:
                    break
            print(f'LINE-FIXED fillcolor at line {i+1}, skipped {skip_next}')
            replaced = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

# Fix empty label warnings - replace st.radio("", with label_visibility
content = content.replace(
    'page = st.radio("Navigate", [',
    'page = st.radio("Navigation Menu", [')
content = content.replace(
    'label_visibility=\'collapsed\')',
    'label_visibility=\'collapsed\')')

# Fix any remaining empty string labels in st.radio
import re
content = re.sub(
    r'st\.radio\("",\s*\[',
    'st.radio("Select option", [',
    content)

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('\nSAVED dashboard.py')

# Verify
with open('dashboard.py', encoding='utf-8') as f:
    check = f.read()
if 'rgba(00d4ff' in check:
    print('WARNING: bad rgba still present')
    # Find it
    for i, line in enumerate(check.split('\n'), 1):
        if 'rgba(00d4ff' in line:
            print(f'  Line {i}: {line.strip()}')
else:
    print('OK: no bad rgba found')
if '_hex_to_rgba' in check:
    print('OK: _hex_to_rgba present')
if 'FILL_COLORS' in check:
    print('OK: FILL_COLORS present')
