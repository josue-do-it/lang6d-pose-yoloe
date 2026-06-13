with open('/workspace/run_yoloe_ycbv_query.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Detect the broken block: "    else:\n        if isinstance..."
    if (line.strip() == 'else:' and
        i + 1 < len(lines) and
        'if isinstance(obj_f, int):' in lines[i+1] and
        i + 2 < len(lines) and
        'YCBV_YOLOE_PROMPTS' in lines[i+2]):
        # Skip these 3 broken lines, replace with correct else branch
        new_lines.append('    else:\n')
        new_lines.append('        prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")\n')
        i += 3  # skip else: + if isinstance + prompt YCBV
        # skip the next else: and HO3D line if present
        while i < len(lines) and lines[i].strip() in ['else:', 'prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")']:
            i += 1
    else:
        new_lines.append(line)
        i += 1

with open('/workspace/run_yoloe_ycbv_query.py', 'w') as f:
    f.writelines(new_lines)
print('Fixed')
