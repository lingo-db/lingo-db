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

def extract_subop_filename(json_file):
    """Extracts the filename from the JSON based on the specified argument."""
    with open(json_file, 'r') as file:
        data = json.load(file)
        for i in range(len(data)):
            if data[i].get("argument") == "subop-prepare-lowering":
                return data[i].get("file")
    raise ValueError(f"No object found with argument {"subop-prepare-lowering"}")

def extract_llvm_filename(json_file):
    with open(json_file,"r") as file:
        data=json.load(file)
        return data[-1].get("file")

def combine_json_files(trace, plan,subopplan, layers,perf, output_file):
    """Combines three JSON files into a single JSON object and writes it to an output file."""
    combined = {
        "fileType": "insights",
        "trace": trace,
        "plan": plan,
        "subopplan": subopplan,
        "layers": layers,
        "perf": perf
    }
    with open(output_file, 'w') as out_file:
        json.dump(combined, out_file, indent=None)

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    # Set environment variables
    env_vars = {
        "LINGODB_EXECUTION_MODE": "SPEED",
        "LINGODB_SNAPSHOT_PASSES": "true",
        "LINGODB_SNAPSHOT_DIR": "./snapshots"
    }

    # Define directories and filenames
    bin_dir = os.getenv('BIN_DIR', './build/lingodb-relwithdebinfo')  # Replace with actual BIN_DIR
    snapshot_json = "./snapshots/detailed-snapshot-info.json"

    # Command 1: Run SQL
    command_1 = f'env {" ".join([f"{k}={v}" for k, v in env_vars.items()])} {bin_dir}/run-sql {query} {data_dir}'
    run_command(command_1)

    # Command 2: Analyze snapshots
    command_2 = f'{bin_dir}/mlir-analyze-snapshots {snapshot_json} mlir-layers.json'
    run_command(command_2)

    # Extract filename from JSON
    extracted_file = extract_relalg_filename(snapshot_json)
    extracted_llvm_filename=extract_llvm_filename(snapshot_json)
    extracted_subop_filename=extract_subop_filename(snapshot_json)

    # Run mlir-db-opt command
    command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_file} {extracted_file}.alt'
    run_command(command_3)
    # Run mlir-db-opt command
    command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_llvm_filename} {extracted_llvm_filename}.alt'
    run_command(command_3)
    command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_subop_filename} {extracted_subop_filename}.alt'
    run_command(command_3)
    perf_env_vars={
        "LINGODB_EXECUTION_MODE": "PERF",
        "LINGODB_BACKEND_ONLY": "true",
        "LINGODB_COMPILATION_LLVM_LOWERING": "false",
        "LINGODB_TRACE_DIR":"/tmp"
    }

    command_perf = f'env {" ".join([f"{k}={v}" for k, v in perf_env_vars.items()])}  {bin_dir}/run-mlir {extracted_llvm_filename}.alt {data_dir}'
    perf_run=run_command(command_perf)
    print(perf_run)

    command_perf_extract_generated=f"perf annotate -n --stdio -l --no-source -d llvm-jit-static.so | python3 {dir_path}/perf-extract-generated.py"
    perf_generated=json.loads(run_command(command_perf_extract_generated))
    command_perf_extract_overview=f"perf report --stdio -n -t \";\" -q | python3 {dir_path}/perf-extract-overview.py"
    perf_overview=json.loads(run_command(command_perf_extract_overview))
    # Run mlir-to-json command
    command_4 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-to-json {extracted_file}.alt {data_dir}'
    plan=json.loads(run_command(command_4))
    command_5 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-subop-to-json {extracted_subop_filename}.alt {data_dir}'
    subop_plan=json.loads(run_command(command_5))
    layers=json.load(open("mlir-layers.json"))
    trace=json.load(open("lingodb.trace"))
    # Combine JSON files
    combine_json_files(trace["trace"], plan,subop_plan, layers,{"overview": perf_overview,"generated":perf_generated}, "insights.json")

if __name__ == "__main__":
    main()