import subprocess
import os
import concurrent.futures  # Import for parallel execution
from tqdm import tqdm  # Import tqdm for the progress bar
import tempfile  # Import tempfile for creating temporary directories
from glob import glob


# Function to process a single query
def process_single_query(q_number, db_directory):
    """
    Executes the old and new commands for a given query number,
    compares their outputs, and returns True for success, False for failure.
    Temporary files are created in a temporary directory.
    """
    sql_file = f'./resources/sql/tpcds/{q_number}.sql'

    # Define environment variables for the new command
    env = dict(**os.environ, LINGODB_PARALLELISM='4', LINGODB_EXECUTION_MODE='BASELINE')

    # Create a temporary directory for the current query's files
    # This directory will be automatically cleaned up when the 'with' block exits
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct full paths for the temporary files within the temp_dir
        old_output_path = os.path.join(temp_dir, f'old_{q_number}.txt')
        new_output_path = os.path.join(temp_dir, f'new_{q_number}.txt')

        # Define commands
        old_cmd = ['cmake-build-debug/run-sql', sql_file, db_directory]
        new_cmd = ['cmake-build-debug/run-sql', sql_file, db_directory]

        try:
            # Run old command and capture output to the temporary file
            with open(old_output_path, 'w') as f_old:
                subprocess.run(old_cmd, stdout=f_old, stderr=subprocess.PIPE, check=True)

            # Run new command and capture output to the temporary file
            with open(new_output_path, 'w') as f_new:
                subprocess.run(new_cmd, stdout=f_new, stderr=subprocess.PIPE, check=True, env=env)

            # Compare outputs using the temporary file paths
            diff = subprocess.run(['diff', old_output_path, new_output_path], stdout=subprocess.PIPE)

            # No need to explicitly remove files; TemporaryDirectory handles cleanup

            if diff.returncode == 0:
                tqdm.write(f"{q_number} \u2713")
                return True  # Success
            else:
                tqdm.write(f"{q_number} \u274c")
                return False  # Failure (diff found differences)
        except subprocess.CalledProcessError as e:
            # Handle errors from subprocess commands (e.g., non-zero exit code)
            # print(f"Subprocess error for query {q_number}: {e}", file=sys.stderr)
            # print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
            return False
        except Exception as e:
            # Catch any other unexpected errors during processing
            # print(f"An unexpected error occurred for query {q_number}: {e}", file=sys.stderr)
            return False


success_count = 0
total_queries = 99
failed_queries = []
db_directory = './resources/data/tpcds-1/'  # Define outside the loop
sql_directory = './resources/sql/tpcds/'  # Define outside the loop

# for all files in the sql_directory, extract the number from {}.sql
query_numbers = [os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(sql_directory, '*.sql'))]
query_numbers.remove('initialize')

# Use ProcessPoolExecutor for parallel execution
# max_workers can be adjusted based on your system's capabilities
# It's generally good to set it to os.cpu_count() or slightly more if tasks are I/O bound
with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Map the process_single_query function to each query number
    # tqdm wraps the executor.map to provide a progress bar for the parallel tasks
    results = list(tqdm(executor.map(process_single_query, range(1, total_queries + 1), [db_directory] * total_queries),
                        total=total_queries,
                        desc="Processing SQL Queries in Parallel",
                        unit="query"))

# After all tasks are complete, aggregate the results
for q_index, result in enumerate(results):
    query_number = q_index + 1  # Queries are 1-indexed
    if result:
        success_count += 1
    else:
        failed_queries.append(query_number)

percent_successful = (success_count / total_queries) * 100
print(f'\n--- Final Report ---')
print(f'Successful: {percent_successful:.2f}%')
print(f'Failed queries: {failed_queries}')
