#!/usr/bin/env python3
"""
Ransomchats Dataset Updater
Manages dataset downloads, versioning, and integrity verification.

Features:
- Automatic download from GitHub with retry logic
- Backup management for previous versions
- Progress bars for download visualization
- Hash verification for integrity checks
- Disk space validation before operations
- Comprehensive statistics and manifest generation

Usage:
    python update_database.py

Author: Brilant Gashi
Project: LLM Ransomware Profiling - University of Brescia
"""

import os
import json
import shutil
import requests
import zipfile
import hashlib
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DatabaseUpdater")

# Optional imports with fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    logger.warning("tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from src.handlers.error_handler import UniBSErrorHandler
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    logger.warning("Error handler not available. Continuing without retry logic.")
    ERROR_HANDLER_AVAILABLE = False


class DatabaseUpdater:
    """
    Manages Ransomchats dataset updates, versioning, and integrity verification.
    
    This class handles the complete lifecycle of dataset management:
    - Downloads latest version from GitHub
    - Creates timestamped backups before updates
    - Extracts and validates downloaded archives
    - Generates unified messages.json from multiple sources
    - Tracks statistics and metadata in manifest file
    - Performs integrity checks and disk space validation
    
    Attributes:
        base_dir (Path): Project root directory
        data_dir (Path): Raw data storage directory
        ransomchats_dir (Path): Extracted dataset directory
        archive_dir (Path): Backup storage directory
        manifest_file (Path): Metadata tracking file
        error_handler (UniBSErrorHandler): Retry logic handler
        config (dict): Configuration from model_config.yaml
    
    Example:
        >>> updater = DatabaseUpdater()
        >>> success = updater.run()
        >>> if success:
        ...     print("Dataset updated successfully!")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database updater with configuration.
        
        Args:
            config_path: Path to model_config.yaml (default: config/model_config.yaml)
        """
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data" / "raw"
        self.ransomchats_dir = self.data_dir / "Ransomchats-main"
        self.archive_dir = self.data_dir / "archive"
        self.manifest_file = self.data_dir / "DATA_MANIFEST.json"
        
        # Create necessary directories
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_path is None:
            config_path = self.base_dir / "config" / "model_config.yaml"
        
        self.config = self._load_config(config_path)
        
        # Initialize error handler if available
        if ERROR_HANDLER_AVAILABLE:
            retry_config = self.config.get('processing', {}).get('retry', {})
            self.error_handler = UniBSErrorHandler(
                max_retries=retry_config.get('max_attempts', 3),
                backoff_factor=retry_config.get('backoff_factor', 2)
            )
        else:
            self.error_handler = None
        
        # Get repository URL from config
        dataset_config = self.config.get('dataset', {})
        self.github_repo = dataset_config.get('github_repo', 'Casualtek/Ransomchats')
        self.zip_url = f"https://github.com/{self.github_repo}/archive/refs/heads/main.zip"
        
        logger.info(f"DatabaseUpdater initialized")
        logger.info(f"Repository: {self.github_repo}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def _load_config(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Configuration dictionary
        """
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def check_disk_space(self, required_mb: int = 100) -> bool:
        """
        Verify sufficient disk space is available.
        
        Args:
            required_mb: Minimum required space in megabytes
        
        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            stat = shutil.disk_usage(self.data_dir)
            free_mb = stat.free // (1024 * 1024)
            
            if free_mb < required_mb:
                logger.error(
                    f"❌ Insufficient disk space: {free_mb}MB available, "
                    f"{required_mb}MB required"
                )
                return False
            
            logger.info(f"✅ Disk space OK: {free_mb}MB available")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Could not check disk space: {e}")
            return True  # Continue anyway if check fails
    
    def download_ransomchats(self) -> bool:
        """
        Download Ransomchats repository from GitHub with progress bar and retry logic.
        
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"📥 Downloading from {self.zip_url}...")
        
        zip_path = self.data_dir / "Ransomchats-main.zip"
        
        def _perform_download():
            """Inner function for download with retry support."""
            response = requests.get(self.zip_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            if TQDM_AVAILABLE and total_size > 0:
                with open(zip_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                    ncols=80
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Simple download without progress bar
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            return True
        
        try:
            # Use error handler with retry if available
            if self.error_handler:
                self.error_handler.with_retry(_perform_download)
            else:
                _perform_download()
            
            logger.info(f"✅ Download completed: {zip_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def verify_download(self, zip_path: Path) -> Tuple[bool, str]:
        """
        Verify downloaded file integrity using SHA256 hash.
        
        Args:
            zip_path: Path to downloaded zip file
        
        Returns:
            Tuple of (success: bool, hash: str)
        """
        logger.info("🔍 Verifying download integrity...")
        
        try:
            sha256_hash = hashlib.sha256()
            
            with open(zip_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            file_hash = sha256_hash.hexdigest()
            
            # Log hash for verification records
            logger.info(f"✅ SHA256: {file_hash[:32]}...")
            
            # Verify file size is reasonable (> 1KB)
            file_size_mb = zip_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.001:
                logger.error("❌ Downloaded file is suspiciously small")
                return False, file_hash
            
            logger.info(f"✅ File size: {file_size_mb:.2f}MB")
            return True, file_hash
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False, ""
    
    def extract_zip(self) -> bool:
        """
        Extract the downloaded zip file with backup management.
        
        Returns:
            True if extraction successful, False otherwise
        """
        logger.info("📦 Extracting archive...")
        
        zip_path = self.data_dir / "Ransomchats-main.zip"
        
        if not zip_path.exists():
            logger.error(f"❌ File not found: {zip_path}")
            return False
        
        try:
            # Backup existing folder if present
            if self.ransomchats_dir.exists():
                logger.info("💾 Creating backup of previous version...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = self.archive_dir / f"Ransomchats-main_backup_{timestamp}"
                shutil.move(str(self.ransomchats_dir), str(backup_dir))
                logger.info(f"✅ Backup saved to: {backup_dir.name}")
            
            # Extract archive
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total file count for progress bar
                members = zip_ref.namelist()
                
                if TQDM_AVAILABLE:
                    for member in tqdm(members, desc="Extracting", ncols=80):
                        zip_ref.extract(member, self.data_dir)
                else:
                    zip_ref.extractall(self.data_dir)
            
            # Verify extraction
            extracted = self.data_dir / "Ransomchats-main"
            if extracted.exists():
                logger.info(f"✅ Extraction completed: {extracted}")
                return True
            else:
                logger.error("❌ Extracted folder not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return False
    
    def generate_messages_json(self) -> bool:
        """
        Generate unified messages.json from extracted folder structure.
        
        Scans all ransomware group directories, loads individual chat JSON files,
        and consolidates them into a single messages.json file.
        
        Returns:
            True if generation successful, False otherwise
        """
        logger.info("📝 Generating messages.json...")
        
        if not self.ransomchats_dir.exists():
            logger.error(f"❌ Folder not found: {self.ransomchats_dir}")
            return False
        
        all_chats = {}
        
        # Get ignored groups from config
        ignored = set(self.config.get('dataset', {}).get('ignored_groups', [
            '.git', '.github', 'parsers', 'src'
        ]))
        
        try:
            folders = [f for f in self.ransomchats_dir.iterdir() if f.is_dir()]
            
            # Process each ransomware group folder
            for folder in folders:
                if folder.name in ignored or folder.name.startswith('.'):
                    continue
                
                group_name = folder.name
                json_files = list(folder.glob("*.json"))
                
                if not json_files:
                    continue
                
                if group_name not in all_chats:
                    all_chats[group_name] = {}
                
                logger.info(f"  📂 Processing {group_name}: {len(json_files)} chats")
                
                # Process each chat file
                for fpath in json_files:
                    try:
                        content = json.loads(fpath.read_text(encoding='utf-8'))
                        
                        # Handle different JSON structures
                        msgs = []
                        if isinstance(content, list):
                            msgs = content
                        elif isinstance(content, dict):
                            msgs = content.get("messages", content.get("dialogue", []))
                        
                        if msgs:
                            chat_id = fpath.stem
                            all_chats[group_name][chat_id] = {"dialogue": msgs}
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️  Invalid JSON in {fpath.name}: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️  Error reading {fpath.name}: {e}")
            
            if not all_chats:
                logger.error("❌ No chats found in dataset")
                return False
            
            # Save consolidated messages.json
            messages_path = self.data_dir / "messages.json"
            with open(messages_path, 'w', encoding='utf-8') as f:
                json.dump(all_chats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ messages.json generated: {messages_path}")
            logger.info(f"   Total groups: {len(all_chats)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return False
    
    def count_stats(self) -> dict:
        """
        Calculate comprehensive dataset statistics.
        
        Returns:
            Dictionary containing:
            - total_groups: Number of ransomware groups
            - total_chats: Total number of conversations
            - total_messages: Total number of individual messages
            - groups: Per-group breakdown with chat and message counts
        """
        stats = {
            "total_groups": 0,
            "total_chats": 0,
            "total_messages": 0,
            "groups": {}
        }
        
        try:
            messages_path = self.data_dir / "messages.json"
            
            if not messages_path.exists():
                logger.warning("⚠️  messages.json not found, cannot calculate stats")
                return stats
            
            with open(messages_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats["total_groups"] = len(data)
            
            for group_name, chats in data.items():
                group_chat_count = len(chats)
                group_msg_count = sum(
                    len(c.get("dialogue", [])) for c in chats.values()
                )
                
                stats["total_chats"] += group_chat_count
                stats["total_messages"] += group_msg_count
                stats["groups"][group_name] = {
                    "chat_count": group_chat_count,
                    "message_count": group_msg_count
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error counting statistics: {e}")
            return stats
    
    def update_manifest(self, stats: dict, file_hash: str = "") -> bool:
        """
        Update manifest file with dataset metadata and statistics.
        
        Args:
            stats: Statistics dictionary from count_stats()
            file_hash: SHA256 hash of downloaded archive
        
        Returns:
            True if manifest updated successfully, False otherwise
        """
        logger.info("📋 Updating manifest...")
        
        manifest = {
            "dataset_name": "Ransomchats Snapshot",
            "last_updated": datetime.now().isoformat(),
            "source_repository": f"https://github.com/{self.github_repo}",
            "file_hash_sha256": file_hash,
            "statistics": stats,
            "config": {
                "ignored_groups": list(self.config.get('dataset', {}).get('ignored_groups', [])),
                "source_type": self.config.get('dataset', {}).get('source', 'local')
            }
        }
        
        try:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Manifest updated: {self.manifest_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Manifest update failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Remove temporary files after successful processing.
        
        Removes the downloaded zip file to save disk space.
        """
        zip_path = self.data_dir / "Ransomchats-main.zip"
        
        try:
            if zip_path.exists():
                zip_path.unlink()
                logger.info("🧹 Temporary zip file removed")
        except Exception as e:
            logger.warning(f"⚠️  Could not remove zip file: {e}")
    
    def run(self) -> bool:
        """
        Execute the complete dataset update workflow.
        
        Workflow:
        1. Check disk space availability
        2. Download latest dataset from GitHub
        3. Verify download integrity
        4. Extract archive (with backup)
        5. Generate unified messages.json
        6. Calculate statistics
        7. Update manifest file
        8. Cleanup temporary files
        
        Returns:
            True if all steps successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("🚀 DATABASE UPDATE STARTED")
        logger.info("=" * 70)
        
        # Step 1: Check disk space
        if not self.check_disk_space(required_mb=100):
            logger.error("❌ Update aborted: insufficient disk space")
            return False
        
        # Step 2: Download dataset
        if not self.download_ransomchats():
            logger.error("❌ Update aborted: download failed")
            return False
        
        # Step 3: Verify integrity
        zip_path = self.data_dir / "Ransomchats-main.zip"
        verified, file_hash = self.verify_download(zip_path)
        if not verified:
            logger.warning("⚠️  Download verification failed, but continuing...")
            file_hash = ""
        
        # Step 4: Extract archive
        if not self.extract_zip():
            logger.error("❌ Update aborted: extraction failed")
            return False
        
        # Step 5: Generate messages.json
        if not self.generate_messages_json():
            logger.error("❌ Update aborted: messages.json generation failed")
            return False
        
        # Step 6: Calculate statistics
        stats = self.count_stats()
        
        # Step 7: Update manifest
        if not self.update_manifest(stats, file_hash):
            logger.warning("⚠️  Manifest update failed, but dataset is ready")
        
        # Step 8: Cleanup
        self.cleanup()
        
        # Print summary
        logger.info("=" * 70)
        logger.info("✅ DATABASE UPDATE COMPLETED SUCCESSFULLY")
        logger.info("-" * 70)
        logger.info(f"📊 Statistics:")
        logger.info(f"   Ransomware Groups: {stats['total_groups']}")
        logger.info(f"   Total Chats:       {stats['total_chats']}")
        logger.info(f"   Total Messages:    {stats['total_messages']}")
        logger.info(f"📂 Location: {self.data_dir / 'messages.json'}")
        logger.info("=" * 70)
        
        # Generate error report if handler available
        if self.error_handler and self.error_handler.get_error_count() > 0:
            logger.info("\n⚠️  Some errors occurred during update:")
            print(self.error_handler.get_error_report(max_entries=5))
        
        return True


def main():
    """Main entry point for command-line execution."""
    try:
        updater = DatabaseUpdater()
        success = updater.run()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Update interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
