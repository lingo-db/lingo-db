import re
import json
import sys


functions = []
current_function = None

for line in sys.stdin:
    # Debug print for each line being processed

    # Match the function header
    func_header_match = re.match(r"^\s*:\s*\d+\s+([0-9a-f]+)\s+<([\w_]+)>:$", line)
    if func_header_match:
        if current_function:
            functions.append(current_function)

        current_function = {
            "func": func_header_match.group(2),
            "assembly": []
        }
        continue

    # Match the assembly line with sample data
    assembly_match = re.match(
        r"^\s*(\d+)\s*:\s*([0-9a-f]+):\s+([a-z0-9\s,%\(\)\$]+)(?:\s+//\s+([\w:\.-]+))?$",
        line
    )
    if assembly_match and current_function:
        samples = int(assembly_match.group(1))
        ip = assembly_match.group(2)
        asm = assembly_match.group(3).strip()
        loc = assembly_match.group(4) if assembly_match.group(4) else None
        if loc:
            file, line=loc.split(":",2)
            file=file.split(".mlir")[0]
            loc=file+":"+line
        current_function["assembly"].append({
            "samples": samples,
            "ip": ip,
            "asm": asm,
            "loc": loc
        })

# Add the last function if any
if current_function:
    functions.append(current_function)

print(json.dumps(functions))