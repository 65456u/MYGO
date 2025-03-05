import os
import argparse

def collect_python_code_to_markdown(directory=".", output_file="python_code.md"):
    """
    Collects all Python code files (.py) in a directory and its subdirectories,
    and writes their paths and content into a Markdown file.

    Args:
        directory (str, optional): The directory to search for Python files.
                                     Defaults to the current directory (".")
        output_file (str, optional): The name of the Markdown file to create.
                                      Defaults to "python_code.md".
    """

    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Python Code Collection\n\n")
        md_file.write("This document contains Python code found in the directory:\n")
        md_file.write(f"`{os.path.abspath(directory)}`\n\n")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as py_file:
                            code_content = py_file.read()

                        md_file.write(f"## File: `{filepath}`\n\n")
                        md_file.write("```python\n")
                        md_file.write(code_content)
                        md_file.write("\n```\n\n")

                    except Exception as e:
                        md_file.write(f"## Error reading file: `{filepath}`\n\n")
                        md_file.write(f"Error: `{e}`\n\n")
                        md_file.write("---\n\n") # Separator for clarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Python code into a Markdown file.")
    parser.add_argument("-d", "--directory", default=".", help="Directory to search for Python files (default: current directory).")
    parser.add_argument("-o", "--output", default="python_code.md", help="Output Markdown file name (default: python_code.md).")

    args = parser.parse_args()

    collect_python_code_to_markdown(args.directory, args.output)
    print(f"Python code collected and saved to '{args.output}' in Markdown format.")