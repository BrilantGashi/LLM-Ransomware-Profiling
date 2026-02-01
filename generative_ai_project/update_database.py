#!/usr/bin/env python3
"""
Script to update the Ransomchats dataset locally.
Usage: python update_database.py
"""

import os
import json
import shutil
import requests
import zipfile
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DatabaseUpdater")


class DatabaseUpdater:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data" / "raw"
        self.ransomchats_dir = self.data_dir / "Ransomchats-main"
        self.archive_dir = self.data_dir / "archive"
        self.manifest_file = self.data_dir / "DATA_MANIFEST.json"
        
        # Create archive folder if it doesn't exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def download_ransomchats(self) -> bool:
        """Download Ransomchats repository from GitHub."""
        logger.info("Downloading Ransomchats repository...")
        
        zip_url = "https://github.com/Casualtek/Ransomchats/archive/refs/heads/main.zip"
        zip_path = self.data_dir / "Ransomchats-main.zip"
        
        try:
            response = requests.get(zip_url, timeout=60)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Download completed: {zip_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def extract_zip(self) -> bool:
        """Extract the zip file."""
        logger.info("Extracting archive...")
        
        zip_path = self.data_dir / "Ransomchats-main.zip"
        
        if not zip_path.exists():
            logger.error(f"File not found: {zip_path}")
            return False
        
        try:
            # Backup old folder if it exists
            if self.ransomchats_dir.exists():
                logger.info("Creating backup of previous version...")
                backup_dir = self.archive_dir / f"Ransomchats-main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(self.ransomchats_dir), str(backup_dir))
                logger.info(f"Backup saved to: {backup_dir}")
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Verify extraction
            extracted = self.data_dir / "Ransomchats-main"
            if extracted.exists():
                logger.info(f"Extraction completed: {extracted}")
                return True
            else:
                logger.error("Extracted folder not found")
                return False
                
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def generate_messages_json(self) -> bool:
        """Generate messages.json from extracted folder."""
        logger.info("Generating messages.json...")
        
        if not self.ransomchats_dir.exists():
            logger.error(f"Folder not found: {self.ransomchats_dir}")
            return False
        
        all_chats = {}
        ignored = {'.git', '.github', 'parsers', 'src'}
        
        try:
            for folder in self.ransomchats_dir.iterdir():
                if not folder.is_dir() or folder.name in ignored or folder.name.startswith('.'):
                    continue
                
                group_name = folder.name
                json_files = list(folder.glob("*.json"))
                
                if not json_files:
                    continue
                
                if group_name not in all_chats:
                    all_chats[group_name] = {}
                
                logger.info(f"Processing {group_name}: {len(json_files)} chats")
                
                for fpath in json_files:
                    try:
                        content = json.loads(fpath.read_text(encoding='utf-8'))
                        
                        msgs = []
                        if isinstance(content, list):
                            msgs = content
                        elif isinstance(content, dict):
                            msgs = content.get("messages", [])
                        
                        if msgs:
                            chat_id = fpath.stem
                            all_chats[group_name][chat_id] = {"dialogue": msgs}
                    except Exception as e:
                        logger.warning(f"Error reading {fpath.name}: {e}")
            
            if not all_chats:
                logger.error("No chats found")
                return False
            
            # Save messages.json
            messages_path = self.data_dir / "messages.json"
            with open(messages_path, 'w', encoding='utf-8') as f:
                json.dump(all_chats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"messages.json generated: {messages_path}")
            return True
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return False
    
    def count_stats(self) -> dict:
        """Count dataset statistics."""
        stats = {
            "total_groups": 0,
            "total_chats": 0,
            "total_messages": 0,
            "groups": {}
        }
        
        try:
            messages_path = self.data_dir / "messages.json"
            with open(messages_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats["total_groups"] = len(data)
            
            for group_name, chats in data.items():
                group_chat_count = len(chats)
                group_msg_count = sum(len(c.get("dialogue", [])) for c in chats.values())
                
                stats["total_chats"] += group_chat_count
                stats["total_messages"] += group_msg_count
                stats["groups"][group_name] = {
                    "chat_count": group_chat_count,
                    "message_count": group_msg_count
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error counting statistics: {e}")
            return stats
    
    def update_manifest(self, stats: dict) -> bool:
        """Update manifest file with metadata."""
        logger.info("Updating manifest...")
        
        manifest = {
            "dataset_name": "Ransomchats Snapshot",
            "last_updated": datetime.now().isoformat(),
            "source_repository": "https://github.com/Casualtek/Ransomchats",
            "statistics": stats
        }
        
        try:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Manifest updated: {self.manifest_file}")
            return True
        except Exception as e:
            logger.error(f"Manifest update failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """Remove zip file after extraction."""
        zip_path = self.data_dir / "Ransomchats-main.zip"
        try:
            if zip_path.exists():
                zip_path.unlink()
                logger.info("Temporary zip file removed")
        except Exception as e:
            logger.warning(f"Warning: Could not remove zip file: {e}")
    
    def run(self) -> bool:
        """Execute the complete update process."""
        logger.info("=" * 70)
        logger.info("DATABASE UPDATE STARTED")
        logger.info("=" * 70)
        
        if not self.download_ransomchats():
            return False
        
        if not self.extract_zip():
            return False
        
        if not self.generate_messages_json():
            return False
        
        stats = self.count_stats()
        
        if not self.update_manifest(stats):
            return False
        
        self.cleanup()
        
        logger.info("=" * 70)
        logger.info("DATABASE UPDATE COMPLETED SUCCESSFULLY")
        logger.info("-" * 70)
        logger.info(f"Groups: {stats['total_groups']}")
        logger.info(f"Chats: {stats['total_chats']}")
        logger.info(f"Messages: {stats['total_messages']}")
        logger.info("=" * 70)
        
        return True


if __name__ == "__main__":
    updater = DatabaseUpdater()
    success = updater.run()
    exit(0 if success else 1)
