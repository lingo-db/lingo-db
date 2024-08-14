import json
import os
import subprocess
import sys
query = sys.argv[1]
data_dir = sys.argv[2]
def run_command(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(e.stdout.decode())
        print(e.stderr.decode())
        raise

def extract_relalg_filename(json_file):
    """Extracts the filename from the JSON based on the specified argument."""
    with open(json_file, 'r') as file:
        data = json.load(file)
        for i in range(len(data)):
            if data[i].get("argument") == "relalg-introduce-tmp":
                return data[i+1].get("file")
    raise ValueError(f"No object found with argument {"relalg-introduce-tmp"}")

def combine_json_files(trace, plan, layers, output_file):
    """Combines three JSON files into a single JSON object and writes it to an output file."""
    combined = {
        "trace": trace,
        "plan": plan,
        "layers": layers
    }
    with open(output_file, 'w') as out_file:
        json.dump(combined, out_file, indent=None)

def main():
    # Set environment variables
    env_vars = {
        "LINGODB_EXECUTION_MODE": "SPEED",
        "LINGODB_SNAPSHOT_PASSES": "true",
        "LINGODB_SNAPSHOT_DIR": "./snapshots"
    }

    # Define directories and filenames
    bin_dir = os.getenv('BIN_DIR', './build/lingodb-relwithdeb')  # Replace with actual BIN_DIR
    snapshot_json = "./snapshots/detailed-snapshot-info.json"

    # Command 1: Run SQL
    command_1 = f'env {" ".join([f"{k}={v}" for k, v in env_vars.items()])} {bin_dir}/run-sql {query} {data_dir}'
    run_command(command_1)

    # Command 2: Analyze snapshots
    command_2 = f'{bin_dir}/mlir-analyze-snapshots {snapshot_json}'
    run_command(command_2)

    # Extract filename from JSON
    extracted_file = extract_relalg_filename(snapshot_json)

    # Run mlir-db-opt command
    command_3 = f'{bin_dir}/mlir-db-opt --strip-debuginfo {extracted_file} > {extracted_file}.alt'
    run_command(command_3)

    # Run mlir-to-json command
    command_4 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-to-json {extracted_file}.alt {data_dir}'
    plan=json.loads(run_command(command_4))
    layers=json.load(open("mlir-layers.json"))
    trace=json.load(open("lingodb.trace"))
    # Combine JSON files
    combine_json_files(trace, plan, layers, "insights.json")

if __name__ == "__main__":
    main()