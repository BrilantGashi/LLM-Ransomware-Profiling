import json
import os
import glob
import requests
from pathlib import Path
from typing import Dict, Any, List

# URL ufficiali del repository Ransomchats
CHAT_INDEX_URL = "https://raw.githubusercontent.com/Casualtek/Ransomchats/main/chat_index.json"
BASE_RAW_URL = "https://raw.githubusercontent.com/Casualtek/Ransomchats/main/"


from pathlib import Path
from typing import Dict, Any

def download_and_load_messages_db(local_path_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Carica i dati locali. Se messages.json non esiste, lo crea leggendo
    la cartella 'Ransomchats-main' che hai già scaricato.
    """
    local_path = Path(local_path_str) # es: data/raw/messages.json
    
    # 1. Se messages.json esiste già ed è pieno, usalo
    if local_path.exists():
        try:
            content = json.loads(local_path.read_text(encoding="utf-8"))
            if content: # Se non è vuoto
                print(f"📂 Caricamento dati locali da {local_path}...")
                return content
        except:
            print("⚠️ messages.json corrotto, lo rigenero...")

    # 2. Se non esiste, lo generiamo dalla tua cartella Ransomchats-main
    print("⚠️ File messages.json non trovato. Lo genero dai file locali...")
    
    # Percorso della cartella Ransomchats-main (relativo a data/raw o assoluto)
    # Modifica questo path se la cartella è altrove!
    base_dir = local_path.parent # data/raw
    source_folder = base_dir / "Ransomchats-main" 
    
    if not source_folder.exists():
        print(f"❌ Errore: Cartella {source_folder} non trovata.")
        print("Assicurati di aver estratto Ransomchats-main dentro data/raw/")
        return {}

    all_chats = {}
    ignored = {'.git', '.github', 'parsers', 'src'}
    
    # Scansiona tutte le cartelle (Akira, Avaddon, ecc)
    for folder in source_folder.iterdir():
        if not folder.is_dir() or folder.name in ignored or folder.name.startswith('.'):
            continue
            
        group_name = folder.name
        json_files = list(folder.glob("*.json"))
        
        if not json_files: continue
        
        if group_name not in all_chats:
            all_chats[group_name] = {}
            
        print(f"   📂 Leggo {group_name}: {len(json_files)} chat...")
        
        for fpath in json_files:
            try:
                content = json.loads(fpath.read_text(encoding='utf-8'))
                
                # Normalizza
                msgs = []
                if isinstance(content, list): msgs = content
                elif isinstance(content, dict): msgs = content.get("messages", [])
                
                if msgs:
                    chat_id = fpath.stem # Nome file senza .json
                    all_chats[group_name][chat_id] = {"dialogue": msgs}
            except:
                pass

    # 3. Salva il risultato
    if all_chats:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(all_chats, f, indent=2)
        print(f"✅ Generato messages.json con successo!")
        return all_chats
    else:
        print("❌ Nessuna chat trovata nelle cartelle locali.")
        return {}
    """
    Carica i dati locali. Se non esistono, li scarica da GitHub gestendo i vari formati possibili.
    """
    local_path = Path(local_path_str)
    
    # Se il file esiste già, lo carichiamo e basta
    if local_path.exists():
        print(f"📂 Caricamento dati locali da {local_path}...")
        return json.loads(local_path.read_text(encoding="utf-8"))
    
    # Se non esiste, scarichiamo
    print("⚠️ File locale non trovato. Avvio download da GitHub...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Scarica l'indice delle chat
        print(f"📥 Scarico indice: {CHAT_INDEX_URL}")
        index_resp = requests.get(CHAT_INDEX_URL)
        index_resp.raise_for_status()
        index = index_resp.json()
        
        print(f"✅ Indice trovato. Totale chat disponibili: {len(index)}")
        
        messages_db = {}
        count = 0
        limit = 5  # Scarichiamo solo 5 chat per test
        
        print(f"⬇️ Scarico le prime {limit} chat...")
        
        for entry in index:
            if count >= limit: break
            
            # --- GESTIONE FORMATI DIVERSI ---
            if isinstance(entry, str):
                # Caso A: L'indice è una lista di stringhe ["folder/file.json"]
                file_path = entry
                parts = file_path.split('/')
                group = parts[0] if len(parts) > 1 else "Unknown"
                chat_id = Path(file_path).stem
            elif isinstance(entry, dict):
                # Caso B: L'indice è una lista di oggetti {"file_path": "..."}
                file_path = entry.get('file_path', '')
                group = entry.get('group', 'Unknown')
                chat_id = entry.get('chat_id', 'Unknown')
            else:
                continue

            # 2. Scarica il file della chat
            file_url = BASE_RAW_URL + file_path
            try:
                r = requests.get(file_url)
                if r.status_code == 200:
                    chat_content = r.json()
                    
                    if group not in messages_db: messages_db[group] = {}
                    
                    # Normalizza i messaggi (lista o dict)
                    msgs = chat_content if isinstance(chat_content, list) else chat_content.get('messages', [])
                    messages_db[group][chat_id] = {"dialogue": msgs}
                    
                    print(f"   🔹 Scaricata: {group}/{chat_id}")
                    count += 1
            except Exception as e:
                print(f"   ❌ Errore chat {file_path}: {e}")
        
        # 3. Salva tutto nel file locale
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(messages_db, f, indent=2)
            
        print(f"✅ Download completato. Salvato in {local_path}")
        return messages_db

    except Exception as e:
        print(f"❌ Errore critico durante il download: {e}")
        # Ritorna dizionario vuoto invece di crashare
        return {}

def clean_message_list(dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pulisce i messaggi da caratteri strani."""
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
