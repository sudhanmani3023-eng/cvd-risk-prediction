import re

with open('main.py', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Find the gender bar chart section - detect the bar call using old x variable
    if '# 6. Gender by class' in line:
        new_lines.append(line)
        i += 1
        # Collect until we finish the legend() call for this block
        block = []
        while i < len(lines):
            block.append(lines[i])
            if 'axes[1, 2].legend()' in lines[i]:
                i += 1
                break
            i += 1
        # Write the fixed block
        indent = '    '
        new_lines.append(indent + 'gen_labels = ["Female (1)", "Male (2)"]\n')
        new_lines.append(indent + 'g0 = df_eda[df_eda["cardio"] == 0]["gender"].value_counts().sort_index()\n')
        new_lines.append(indent + 'g1 = df_eda[df_eda["cardio"] == 1]["gender"].value_counts().sort_index()\n')
        new_lines.append(indent + 'xg = np.arange(2)\n')
        new_lines.append(indent + 'g0_vals = [g0.get(1, 0), g0.get(2, 0)]\n')
        new_lines.append(indent + 'g1_vals = [g1.get(1, 0), g1.get(2, 0)]\n')
        new_lines.append(indent + 'axes[1, 2].bar(xg - w/2, g0_vals, w, color="#2196F3", label="No CVD", edgecolor="white")\n')
        new_lines.append(indent + 'axes[1, 2].bar(xg + w/2, g1_vals, w, color="#FF5722", label="CVD", edgecolor="white")\n')
        new_lines.append(indent + 'axes[1, 2].set_xticks(xg)\n')
        new_lines.append(indent + 'axes[1, 2].set_xticklabels(gen_labels)\n')
        new_lines.append(indent + 'axes[1, 2].set_title("Gender Distribution by CVD Status", fontweight="bold")\n')
        new_lines.append(indent + 'axes[1, 2].set_ylabel("Count")\n')
        new_lines.append(indent + 'axes[1, 2].legend()\n')
    else:
        new_lines.append(line)
        i += 1

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('DONE: main.py patched successfully')

# Verify the fix
with open('main.py', encoding='utf-8') as f:
    content = f.read()
if 'xg = np.arange(2)' in content:
    print('VERIFIED: xg fix is present in main.py')
else:
    print('ERROR: fix was not applied')
