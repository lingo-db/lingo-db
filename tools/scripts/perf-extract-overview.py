import json
from collections import defaultdict
import sys
# File input

# Initialize data structures
output_data = []
unknown_user_samples = 0
unknown_kernel_samples = 0

# Read and process the file
for line in sys.stdin:
    # Split the line by `;` separator and strip whitespace
    parts = [p.strip() for p in line.split(";")]
    if len(parts) < 5:
        continue

    # Extract relevant fields
    samples = int(parts[1].replace(",", "").strip())
    file_name = parts[3]
    symbol = parts[4]

    # Check if the symbol is "unknown"
    if "[unknown]" in file_name and "[k]" in symbol:
        unknown_kernel_samples += samples
    elif "[unknown]" in file_name and "[.]" in symbol:
        unknown_user_samples += samples
    else:
        # Add known symbol entries
        output_data.append({
            "samples": samples,
            "file": file_name,
            "symbol": symbol[4:]  # Extract actual symbol
        })

# Add aggregated unknown user-space symbols
output_data.append({
    "samples": unknown_user_samples,
    "file": "unknown",
    "symbol": "unknown"
})

# Add aggregated unknown kernel symbols
output_data.append({
    "samples": unknown_kernel_samples,
    "file": "kernel",
    "symbol": "unknown"
})

# Output JSON
print(json.dumps(output_data))