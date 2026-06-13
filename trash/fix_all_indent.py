import ast, sys

path = '/workspace/run_yoloe_ycbv_query.py'

with open(path, 'r') as f:
    content = f.read()

# Test syntax first
try:
    ast.parse(content)
    print("File already valid Python — no fix needed")
    sys.exit(0)
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")

lines = content.splitlines(keepends=True)
print(f"Total lines: {len(lines)}")

# Show context around error
err_line = e.lineno - 1
print("Context:")
for i in range(max(0, err_line-3), min(len(lines), err_line+3)):
    print(f"  {i+1:4}: {repr(lines[i])}")

# Fix: find all "else:\n    if isinstance" duplicate pattern
new_lines = []
i = 0
skip_next = 0
while i < len(lines):
    if skip_next > 0:
        skip_next -= 1
        i += 1
        continue
    line = lines[i]
    # Pattern: "    else:\n" followed by "        if isinstance(obj_f, int):\n" 
    # followed by "        prompt = YCBV..." (bad indent)
    if (i + 2 < len(lines) and
        line.strip() == 'else:' and
        'if isinstance' in lines[i+1] and
        'prompt' in lines[i+2] and
        not lines[i+2].startswith('        ')):  # bad indent
        # Replace with correct else branch
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * indent + 'else:\n')
        new_lines.append(' ' * indent + '    prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")\n')
        # Skip the malformed lines
        i += 3
        # Skip any further duplicate else/prompt lines
        while i < len(lines) and lines[i].strip() in ['else:', 'prompt = HO3D_YOLOE_PROMPTS.get(obj_f, "object")']:
            i += 1
        continue
    new_lines.append(line)
    i += 1

new_content = ''.join(new_lines)

# Verify
try:
    ast.parse(new_content)
    with open(path, 'w') as f:
        f.write(new_content)
    print("Fixed and verified OK")
except SyntaxError as e2:
    print(f"Still broken at line {e2.lineno}: {e2.msg}")
    # Show remaining context
    lines2 = new_content.splitlines()
    for i in range(max(0, e2.lineno-3), min(len(lines2), e2.lineno+3)):
        print(f"  {i+1:4}: {repr(lines2[i])}")
