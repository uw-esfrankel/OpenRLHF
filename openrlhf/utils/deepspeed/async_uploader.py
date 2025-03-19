import os
import time
import signal
import subprocess
import json
import shutil
from pathlib import Path
import fcntl
import multiprocessing
from datetime import datetime
import torch
from torch import distributed as dist


class AsyncHFUploader:
    """Manages asynchronous uploads to HuggingFace Hub while training continues."""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.lock_file = self.base_dir / "upload_lock.json"
        self.upload_queue_dir = self.base_dir / "upload_queue"
        self.upload_queue_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the lock file if it doesn't exist
        if not self.lock_file.exists():
            self._write_lock_file({"active_uploads": {}, "upload_queue": []})
    
    def _write_lock_file(self, data):
        """Write to the lock file with proper locking."""
        with open(self.lock_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
    
    def _read_lock_file(self):
        """Read from the lock file with proper locking."""
        if not self.lock_file.exists():
            return {"active_uploads": {}, "upload_queue": []}
        
        with open(self.lock_file, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    
    def _clean_stale_processes(self):
        """Remove processes that no longer exist."""
        data = self._read_lock_file()
        active_uploads = data["active_uploads"]
        
        cleaned = {}
        for upload_id, info in active_uploads.items():
            pid = info.get("pid")
            # Check if process is still running
            try:
                if pid and os.kill(pid, 0) is None:
                    # Process is still running
                    cleaned[upload_id] = info
                else:
                    print(f"Removing stale upload process {upload_id} with PID {pid}")
            except OSError:
                # Process doesn't exist
                print(f"Removing stale upload process {upload_id} with PID {pid}")
                
        if len(cleaned) != len(active_uploads):
            data["active_uploads"] = cleaned
            self._write_lock_file(data)
    
    def queue_upload(self, model_dir, repo_name, revision=None, priority=0):
        """Add an upload to the queue."""
        # Generate a unique ID for this upload
        upload_id = f"upload_{int(time.time())}_{os.getpid()}"
        model_queue_dir = self.upload_queue_dir / upload_id
        
        # Copy the model files to the queue directory
        shutil.copytree(model_dir, model_queue_dir)
        
        # Update the lock file
        data = self._read_lock_file()
        
        # Add to queue with metadata
        queue_entry = {
            "id": upload_id,
            "repo_name": repo_name,
            "model_dir": str(model_queue_dir),
            "revision": revision,
            "created_at": datetime.now().isoformat(),
            "priority": priority  # Higher priority items will be processed first
        }
        
        data["upload_queue"].append(queue_entry)
        # Sort the queue by priority (highest first)
        data["upload_queue"].sort(key=lambda x: x["priority"], reverse=True)
        
        self._write_lock_file(data)
        
        # Try to process the queue
        self.process_queue()
        
        return upload_id
    
    def process_queue(self):
        """Process the upload queue if possible."""
        # Clean up any stale processes first
        self._clean_stale_processes()
        
        # Read current state
        data = self._read_lock_file()
        
        # If there are no items in the queue, nothing to do
        if not data["upload_queue"]:
            return
        
        # Check if we're already at the maximum allowed concurrent uploads
        # (For safety, limit to a reasonable number like 2)
        MAX_CONCURRENT_UPLOADS = 2
        if len(data["active_uploads"]) >= MAX_CONCURRENT_UPLOADS:
            return
        
        # Get the highest priority item from the queue
        upload_item = data["upload_queue"].pop(0)
        
        # Start a new process for this upload
        process = multiprocessing.Process(
            target=self._run_upload_process,
            args=(upload_item["id"], upload_item["model_dir"], 
                  upload_item["repo_name"], upload_item["revision"])
        )
        process.start()
        
        # Record the active upload
        data["active_uploads"][upload_item["id"]] = {
            "pid": process.pid,
            "started_at": datetime.now().isoformat(),
            "model_dir": upload_item["model_dir"],
            "repo_name": upload_item["repo_name"],
            "revision": upload_item["revision"]
        }
        
        self._write_lock_file(data)
    
    def _run_upload_process(self, upload_id, model_dir, repo_name, revision):
        """Function that runs in a separate process to handle the upload."""
        try:
            # Import here to avoid requiring huggingface_hub in the main process
            from huggingface_hub import HfApi
            
            print(f"[PID {os.getpid()}] Starting upload to {repo_name} for {model_dir}")
            api = HfApi()
            
            # Check if repo exists, create if it doesn't
            if not api.repo_exists(repo_name):
                api.create_repo(repo_name, repo_type="model")
            
            # Upload to main branch
            api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_name,
                repo_type="model",
            )
            
            # Create a tag for this version
            if revision:
                api.create_tag(
                    repo_id=repo_name,
                    tag=revision,
                    repo_type="model",
                )
            
            print(f"[PID {os.getpid()}] Successfully uploaded model to {repo_name}" + 
                  (f" with tag {revision}" if revision else ""))
            
        except Exception as e:
            print(f"[PID {os.getpid()}] Error uploading to Hugging Face Hub: {e}")
        
        finally:
            # Clean up by removing this upload from active_uploads
            try:
                data = self._read_lock_file()
                if upload_id in data["active_uploads"]:
                    del data["active_uploads"][upload_id]
                    self._write_lock_file(data)
                
                # Clean up the copied directory
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    
                # Process the next item in the queue if any
                next_data = self._read_lock_file()
                if next_data["upload_queue"]:
                    # Re-run the process_queue function in a new process
                    # to avoid recursion issues
                    uploader = AsyncHFUploader(self.base_dir)
                    uploader.process_queue()
                    
            except Exception as e:
                print(f"[PID {os.getpid()}] Error during cleanup: {e}")