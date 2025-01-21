import json
import os
import re
import subprocess
import sys

query = sys.argv[1]
data_dir = sys.argv[2]

version = "0.0.3"


def extract_error_messages(input_text):
    # Regular expression to match the error format
    error_pattern = re.compile(
        r'loc\("\./snapshots/(?P<filename>[^"]+?)":(?P<line>\d+):\d+\): error: (?P<message>.+)'
    )

    # Extract matches
    matches = error_pattern.finditer(input_text)

    # Create a list of tuples (filename, line, message)
    errors = [
        ((match.group("filename"), int(match.group("line"))), match.group("message"))
        for match in matches
    ]

    return errors


def run_command(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode().strip()

def run_command_with_error(command):
    """Runs a shell command and returns the output."""
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode().strip(), result.stderr.decode().strip()


def extract_relalg_filename(json_file):
    """Extracts the filename from the JSON based on the specified argument."""
    with open(json_file, 'r') as file:
        data = json.load(file)
        for i in range(len(data)):
            if data[i].get("argument") == "relalg-introduce-tmp":
                return data[i + 1].get("file")
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
    with open(json_file, "r") as file:
        data = json.load(file)
        return data[-1].get("file")


dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    ct_mode = "PROFILING"
    # Set environment variables
    env_vars = {
        "LINGODB_EXECUTION_MODE": "SPEED",
        "LINGODB_SNAPSHOT_PASSES": "true",
        "LINGODB_SNAPSHOT_DIR": "./snapshots"
    }

    # Define directories and filenames
    bin_dir = os.getenv('BIN_DIR', './build/lingodb-relwithdebinfo')  # Replace with actual BIN_DIR
    snapshot_json = "./snapshots/detailed-snapshot-info.json"
    with open(query, 'r') as query_file:
        query_sql = query_file.read()

    # Command 1: Run SQL
    command_1 = f'env {" ".join([f"{k}={v}" for k, v in env_vars.items()])} {bin_dir}/run-sql {query} {data_dir}'
    error_list = []
    try:
        run_command(command_1)
    except subprocess.CalledProcessError as e:
        print("Error when running query, switching to debug mode")
        print("STDOUT:")
        print(e.stdout.decode())
        print("STDERR:")
        print(e.stderr.decode())

        # check if snaphots are present by checking if ./snapshots/detailed-snapshot-info.json exists
        if not os.path.exists(snapshot_json):
            print("No snapshots found, exiting")
            sys.exit(1)
        ct_mode = "DEBUG"
        error_list+=extract_error_messages(e.stderr.decode())

    # Command 2: Analyze snapshots
    command_2 = f'{bin_dir}/mlir-analyze-snapshots {snapshot_json} mlir-layers.json'
    analyze_stdout, analyze_stderr = run_command_with_error(command_2)
    layers = json.load(open("mlir-layers.json"))
    passes = json.load(open(snapshot_json))
    if (len(layers) != len(passes)):
        print("When analyzing snapshots, some snapshots could not be analyzed. Switching to debug mode")
        print("STDOUT:")
        print(analyze_stdout)
        print("STDERR:")
        print(analyze_stderr)
        ct_mode = "DEBUG"
        error_list+=extract_error_messages(analyze_stderr)
    if ct_mode == "DEBUG":
        print("Debug mode, skipping profiling")
        plan=None
        subop_plan=None
        try:
            extracted_file = extract_relalg_filename(snapshot_json)
            command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_file} {extracted_file}.alt'
            run_command(command_3)
            command_4 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-to-json {extracted_file}.alt'
            plan = json.loads(run_command(command_4))
        except Exception as e:
            print("Could not generate plan", e)
        try:
            extracted_subop_filename = extract_subop_filename(snapshot_json)
            command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_subop_filename} {extracted_subop_filename}.alt'
            run_command(command_3)
            command_5 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-subop-to-json {extracted_subop_filename}.alt {data_dir}'
            subop_plan,_ = json.loads(run_command(command_5))
        except Exception as e:
            print("Could not generate subop plan", e)
        print("Error list:")
        print(error_list)
        combined = {
            "fileType": "debugging",
            "sql": query_sql,
            "layers": layers,
            "errors": error_list,
            "version": version
            #    "perf": {"overview": perf_overview, "generated": perf_generated}
        }
        if plan is not None:
            combined["plan"] = plan
        if subop_plan is not None:
            combined["subopplan"] = subop_plan
        with open("ct.json", 'w') as out_file:
            json.dump(combined, out_file, indent=None)

    else:
        # Extract filename from JSON
        extracted_file = extract_relalg_filename(snapshot_json)
        extracted_llvm_filename = extract_llvm_filename(snapshot_json)
        extracted_subop_filename = extract_subop_filename(snapshot_json)

        # Run mlir-db-opt command
        command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_file} {extracted_file}.alt'
        run_command(command_3)
        # Run mlir-db-opt command
        command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_llvm_filename} {extracted_llvm_filename}.alt'
        run_command(command_3)
        command_3 = f'bash {dir_path}/clean-snapshot.sh {bin_dir} {extracted_subop_filename} {extracted_subop_filename}.alt'
        run_command(command_3)
        perf_env_vars = {
            "LINGODB_EXECUTION_MODE": "PERF",
            "LINGODB_BACKEND_ONLY": "true",
            "LINGODB_COMPILATION_LLVM_LOWERING": "false",
            "LINGODB_TRACE_DIR": "/tmp"
        }

        command_perf = f'env {" ".join([f"{k}={v}" for k, v in perf_env_vars.items()])}  {bin_dir}/run-mlir {extracted_llvm_filename}.alt {data_dir}'
        perf_run = run_command(command_perf)
        print(perf_run)

        command_perf_extract_generated = f"perf annotate -n --stdio -l --no-source -d llvm-jit-static.so | python3 {dir_path}/perf-extract-generated.py"
        perf_generated = json.loads(run_command(command_perf_extract_generated))
        command_perf_extract_overview = f"perf report --stdio -n -t \";\" -q | python3 {dir_path}/perf-extract-overview.py"
        perf_overview = json.loads(run_command(command_perf_extract_overview))
        # Run mlir-to-json command
        command_4 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-to-json {extracted_file}.alt {data_dir}'
        plan = json.loads(run_command(command_4))
        command_5 = f'env LINGODB_TRACE_DIR=/tmp {bin_dir}/mlir-subop-to-json {extracted_subop_filename}.alt {data_dir}'
        subop_plan = json.loads(run_command(command_5))
        trace = json.load(open("lingodb.trace"))
        combined = {
            "fileType": "profiling",
            "sql": query_sql,
            "trace": trace["trace"],
            "plan": plan,
            "subopplan": subop_plan,
            "layers": layers,
            "perf": {"overview": perf_overview, "generated": perf_generated},
            "version": version
        }
        with open("ct.json", 'w') as out_file:
            json.dump(combined, out_file, indent=None)


if __name__ == "__main__":
    main()
