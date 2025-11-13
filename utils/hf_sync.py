"""Utilities for syncing models from HuggingFace."""

import logging
from pathlib import Path
from typing import Tuple, Optional
from huggingface_hub import hf_hub_download, list_repo_files
import os

log = logging.getLogger(__name__)


def sync_hf_models(
    repo_id: str,
    patterns: Tuple[str, ...] = ("*.pkl",),
    prune: bool = False,
    local_dir: Optional[str] = None
) -> None:
    """Sync models from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/repo")
        patterns: Tuple of file patterns to download (e.g., ("*.pkl", "*.json"))
        prune: If True, remove local files not in the remote repository
        local_dir: Local directory to save models (defaults to "models/")
    """
    if local_dir is None:
        local_dir = "models"
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # List files in the repository
        repo_files = list_repo_files(repo_id, repo_type="model")
        
        # Filter files by patterns
        files_to_download = []
        log.info(f"Repo files: {repo_files}")
        for pattern in patterns:
            # Convert glob pattern to simple extension check
            if pattern.startswith("*."):
                ext = pattern[1:]  # Remove "*"
                files_to_download.extend([f for f in repo_files if f.endswith(ext)])
        
        # Remove duplicates
        files_to_download = list(set(files_to_download))
        
        if not files_to_download:
            log.info(f"No files matching patterns {patterns} found in repository {repo_id}")
            return
        
        log.info(f"Found {len(files_to_download)} file(s) to sync from {repo_id}")
        
        # Download each file
        for file_path in files_to_download:
            try:
                # Get relative path for local storage
                relative_path = file_path
                local_file_path = local_path / relative_path
                
                # Create parent directories if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                    repo_type="model"
                )
                
                log.info(f"Downloaded: {file_path} -> {downloaded_path}")
                
            except Exception as e:
                log.error(f"Failed to download {file_path}: {e}")
                continue
        
        # Prune local files not in remote (if requested)
        if prune:
            # Get all local .pkl files
            local_files = set(local_path.rglob("*.pkl"))
            remote_files = set(files_to_download)
            
            # Find files to remove
            files_to_remove = []
            for local_file in local_files:
                relative = str(local_file.relative_to(local_path)).replace("\\", "/")
                if relative not in remote_files:
                    files_to_remove.append(local_file)
            
            # Remove files
            for file_to_remove in files_to_remove:
                try:
                    file_to_remove.unlink()
                    log.info(f"Pruned local file: {file_to_remove}")
                except Exception as e:
                    log.error(f"Failed to prune {file_to_remove}: {e}")
        
        log.info(f"Sync completed: {len(files_to_download)} file(s) processed")
        
    except Exception as e:
        log.error(f"Error syncing models from {repo_id}: {e}")
        # Don't raise - allow the app to continue even if sync fails
        log.warning("Continuing without model sync")

