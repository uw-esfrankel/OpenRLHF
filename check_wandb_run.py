import wandb
import argparse
import os
import sys

def check_run_finished(project_path, run_name):
    # Initialize the API
    api = wandb.Api(api_key=os.environ["WANDB_TOKEN"])
    
    try:
        # Search for runs with the specific name in the project
        runs = api.runs(project_path, {"display_name": run_name})
        
        # Check if any runs were found
        if len(list(runs)) == 0:
            print(f"No runs found with name '{run_name}' in project '{project_path}'")
            return False
        
        # Check if any of the matching runs are finished
        for run in runs:
            if run.state == "finished":
                print(f"Run '{run_name}' (ID: {run.id}) is finished")
                return True
        
        # If we got here, runs were found but none are finished
        print(f"Run '{run_name}' exists but is not finished (current state: {run.state})")
        return False
        
    except Exception as e:
        print(f"Error checking run status: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check if a specific W&B run is finished")
    parser.add_argument("--project", required=True, help="Project path in format 'username/project_name'")
    parser.add_argument("--name", required=True, help="Run name to check")
    args = parser.parse_args()
    
    # Check if the run is finished
    is_finished = check_run_finished(args.project, args.name)
    
    # Exit with appropriate status code (0 for finished, 1 for not finished)
    sys.exit(0 if is_finished else 1)