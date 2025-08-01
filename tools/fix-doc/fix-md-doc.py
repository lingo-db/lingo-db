import re
import sys

def convert_markdown_to_html(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as file:
        raw_lines = file.readlines()
    # Make sure that {{% markdown %}} and {{% /markdown %}} are not in the same line, split them if necessary
    # also: nothing should be after {{% markdown %}} and before {{% /markdown %}} in the same line
    fixed_lines = []
    for line in raw_lines:
        if "{{% markdown %}}" in line and "{{% /markdown %}}" in line:
            parts = line.split("{{% markdown %}}")
            fixed_lines.append(parts[0] + "{{% markdown %}}")
            for part in parts[1:]:
                if "{{% /markdown %}}" in part:
                    sub_parts = part.split("{{% /markdown %}}")
                    fixed_lines.append("{{% markdown %}}".join(sub_parts[:-1]))
                    fixed_lines.append("{{% /markdown %}}" + sub_parts[-1])
                else:
                    fixed_lines.append(part)
        else:
            fixed_lines.append(line)
    in_markdown_block = False
    output_lines = []
    current_block = []

    # Iterate through each line in the input file
    for line in fixed_lines:
        if line.strip() == "[TOC]":
            # Skip the [TOC] line
            continue
        # Check if we are entering a markdown block
        if "{{% markdown %}}" in line:
            in_markdown_block = True
            # Retain the line before the block and start processing inside the block
            output_lines.append(line.split("{{% markdown %}}")[0])  # Retain text before the marker
            current_block = []  # Reset the current block for processing
            continue

        # Check if we are leaving a markdown block
        if "{{% /markdown %}}" in line:
            in_markdown_block = False
            # Process the block and wrap it in <ul>
            output_lines.append("<ul>")  # Start the unordered list
            # Add processed markdown content (list items)
            for list_item in current_block:
                # Process each list item inside the markdown block
                match = re.match(r'([^\(]+)\s?\(`([^`]+)`\)', list_item.strip())
                if match:
                    attribute = match.group(1).strip()
                    mlir_type = match.group(2).strip()
                    html_item = f"<li><strong>{attribute}</strong>: Enum case: {attribute} ({mlir_type})</li>"
                    output_lines.append(html_item)
                else:
                    output_lines.append(f"<li>{list_item.strip()}</li>")
            output_lines.append("</ul>")  # Close the unordered list
            output_lines.append(line.split("{{% /markdown %}}")[1])  # Retain text after the marker
            continue

        # Process lines inside the markdown block (for list items)
        if in_markdown_block:
            if line.strip().startswith("* "):  # Markdown list item
                current_block.append(line.strip()[2:])  # Remove "* " from the start
            else:
                current_block.append(line)  # Add non-list content to the current block
        else:
            output_lines.append(line)
    # Write the output to the output file
    with open(output_file, 'w') as file:
        file.writelines(output_lines)

    print(f"HTML output written to {output_file}")

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix-md-doc.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_markdown_to_html(input_file, output_file)
