import json
import os
import glob
import requests
from pathlib import Path
from typing import Dict, Any, List


# Official URLs for Ransomchats repository
CHAT_INDEX_URL = "https://raw.githubusercontent.com/Casualtek/Ransomchats/main/chat_index.json"
BASE_RAW_URL = "https://raw.githubusercontent.com/Casualtek/Ransomchats/main/"


def download_and_load_messages_db(local_path_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Load local data. If messages.json does not exist, create it by reading
    the 'Ransomchats-main' folder that has already been downloaded.
    """
    local_path = Path(local_path_str)
    
    # If messages.json exists and is not empty, use it
    if local_path.exists():
        try:
            content = json.loads(local_path.read_text(encoding="utf-8"))
            if content:
                print(f"Loading local data from {local_path}...")
                return content
        except:
            print("messages.json corrupted, regenerating...")

    # If it does not exist, generate it from Ransomchats-main folder
    print("messages.json file not found. Generating from local files...")
    
    # Path to Ransomchats-main folder (relative to data/raw or absolute)
    base_dir = local_path.parent
    source_folder = base_dir / "Ransomchats-main" 
    
    if not source_folder.exists():
        print(f"Error: Folder {source_folder} not found.")
        print("Make sure you have extracted Ransomchats-main inside data/raw/")
        return {}

    all_chats = {}
    ignored = {'.git', '.github', 'parsers', 'src'}
    
    # Scan all folders (Akira, Avaddon, etc)
    for folder in source_folder.iterdir():
        if not folder.is_dir() or folder.name in ignored or folder.name.startswith('.'):
            continue
            
        group_name = folder.name
        json_files = list(folder.glob("*.json"))
        
        if not json_files:
            continue
        
        if group_name not in all_chats:
            all_chats[group_name] = {}
            
        print(f"   Reading {group_name}: {len(json_files)} chats...")
        
        for fpath in json_files:
            try:
                content = json.loads(fpath.read_text(encoding='utf-8'))
                
                # Normalize
                msgs = []
                if isinstance(content, list):
                    msgs = content
                elif isinstance(content, dict):
                    msgs = content.get("messages", [])
                
                if msgs:
                    chat_id = fpath.stem
                    all_chats[group_name][chat_id] = {"dialogue": msgs}
            except:
                pass

    # Save the result
    if all_chats:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(all_chats, f, indent=2)
        print("Generated messages.json successfully!")
        return all_chats
    else:
        print("No chats found in local folders.")
        return {}


def download_from_github(local_path_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Download data from GitHub if local file does not exist.
    Handles various possible data formats.
    """
    local_path = Path(local_path_str)
    
    # If file already exists, load it
    if local_path.exists():
        print(f"Loading local data from {local_path}...")
        return json.loads(local_path.read_text(encoding="utf-8"))
    
    # If it does not exist, download from GitHub
    print("Local file not found. Starting download from GitHub...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download chat index
        print(f"Downloading index: {CHAT_INDEX_URL}")
        index_resp = requests.get(CHAT_INDEX_URL)
        index_resp.raise_for_status()
        index = index_resp.json()
        
        print(f"Index found. Total available chats: {len(index)}")
        
        messages_db = {}
        count = 0
        limit = 5
        
        print(f"Downloading first {limit} chats...")
        
        for entry in index:
            if count >= limit:
                break
            
            # Handle different formats
            if isinstance(entry, str):
                # Case A: Index is a list of strings ["folder/file.json"]
                file_path = entry
                parts = file_path.split('/')
                group = parts[0] if len(parts) > 1 else "Unknown"
                chat_id = Path(file_path).stem
            elif isinstance(entry, dict):
                # Case B: Index is a list of objects {"file_path": "..."}
                file_path = entry.get('file_path', '')
                group = entry.get('group', 'Unknown')
                chat_id = entry.get('chat_id', 'Unknown')
            else:
                continue

            # Download chat file
            file_url = BASE_RAW_URL + file_path
            try:
                r = requests.get(file_url)
                if r.status_code == 200:
                    chat_content = r.json()
                    
                    if group not in messages_db:
                        messages_db[group] = {}
                    
                    # Normalize messages (list or dict)
                    msgs = chat_content if isinstance(chat_content, list) else chat_content.get('messages', [])
                    messages_db[group][chat_id] = {"dialogue": msgs}
                    
                    print(f"   Downloaded: {group}/{chat_id}")
                    count += 1
            except Exception as e:
                print(f"   Error downloading chat {file_path}: {e}")
        
        # Save to local file
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(messages_db, f, indent=2)
            
        print(f"Download complete. Saved to {local_path}")
        return messages_db

    except Exception as e:
        print(f"Critical error during download: {e}")
        return {}


def clean_message_list(dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean messages from special characters and encoding issues."""
    cleaned = []
    for item in dialogue:
        item2 = dict(item)
        msg = item2.get("message", "")
        if isinstance(msg, str):
            try:
                item2["message"] = msg.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass 
        cleaned.append(item2)
    return cleaned
