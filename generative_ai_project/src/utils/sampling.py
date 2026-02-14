"""
Sampling utilities for stratified chat selection based on message count distribution.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def stratified_sample_chats(
    messages_db: Dict,
    random_seed: int = 42,
    verbose: bool = True
) -> List[Tuple[str, str, int]]:
    """
    Campionamento stratificato basato sulla lunghezza delle chat.
    
    Args:
        messages_db: Dictionary dal messages.json {group: {chat_id: {dialogue: [...]}}}
        random_seed: Seed per riproducibilità
        verbose: Stampa statistiche
    
    Returns:
        List di tuple (group, chat_id, message_count)
    """
    random.seed(random_seed)
    
    # Categorizza chat per lunghezza
    bins = {
        '<10': [],
        '10-30': [],
        '30-60': [],
        '60-100': [],
        '100-150': [],
        '>150': []
    }
    
    for group, chats in messages_db.items():
        for chat_id, chat_data in chats.items():
            msg_count = len(chat_data.get('dialogue', []))
            
            if msg_count < 10:
                bins['<10'].append((group, chat_id, msg_count))
            elif 10 <= msg_count <= 30:
                bins['10-30'].append((group, chat_id, msg_count))
            elif 30 < msg_count <= 60:
                bins['30-60'].append((group, chat_id, msg_count))
            elif 60 < msg_count <= 100:
                bins['60-100'].append((group, chat_id, msg_count))
            elif 100 < msg_count <= 150:
                bins['100-150'].append((group, chat_id, msg_count))
            else:
                bins['>150'].append((group, chat_id, msg_count))
    
    if verbose:
        print("\n" + "="*50)
        print("DISTRIBUZIONE DATASET")
        print("="*50)
        for bin_name, chats in bins.items():
            print(f"{bin_name:>10}: {len(chats):>3} chat")
        total = sum(len(c) for c in bins.values())
        print(f"{'TOTALE':>10}: {total:>3} chat")
        print("="*50 + "\n")
    
    # Campionamento stratificato
    sampled = []
    
    if verbose:
        print("CAMPIONAMENTO STRATIFICATO")
        print("-"*50)
    
    # SKIP chat <10 messaggi
    if verbose:
        print(f"❌ Escluse chat <10 messaggi: {len(bins['<10'])}")
    
    # Campiona 10-30: ~55%
    sample_10_30 = random.sample(bins['10-30'], k=min(50, len(bins['10-30'])))
    sampled.extend(sample_10_30)
    if verbose:
        print(f"✓ Campionate 10-30: {len(sample_10_30)}/{len(bins['10-30'])}")
    
    # Campiona 30-60: ~42%
    sample_30_60 = random.sample(bins['30-60'], k=min(25, len(bins['30-60'])))
    sampled.extend(sample_30_60)
    if verbose:
        print(f"✓ Campionate 30-60: {len(sample_30_60)}/{len(bins['30-60'])}")
    
    # Campiona 60-100: ~33%
    sample_60_100 = random.sample(bins['60-100'], k=min(10, len(bins['60-100'])))
    sampled.extend(sample_60_100)
    if verbose:
        print(f"✓ Campionate 60-100: {len(sample_60_100)}/{len(bins['60-100'])}")
    
    # Campiona 100-150: ~30%
    sample_100_150 = random.sample(bins['100-150'], k=min(3, len(bins['100-150'])))
    sampled.extend(sample_100_150)
    if verbose:
        print(f"✓ Campionate 100-150: {len(sample_100_150)}/{len(bins['100-150'])}")
    
    # Campiona >150: le 2 più lunghe + 1 casuale
    if len(bins['>150']) > 0:
        sorted_long = sorted(bins['>150'], key=lambda x: x[2], reverse=True)
        sample_150_plus = sorted_long[:2]
        if len(sorted_long) > 2:
            sample_150_plus.append(random.choice(sorted_long[2:]))
        sampled.extend(sample_150_plus)
        max_msg = max(x[2] for x in bins['>150'])
        if verbose:
            print(f"✓ Campionate >150: {len(sample_150_plus)}/{len(bins['>150'])} (max: {max_msg} msg)")
    
    total_available = sum(len(c) for c in bins.values() if c != bins['<10'])
    
    if verbose:
        print("-"*50)
        print(f"TOTALE CAMPIONATO: {len(sampled)} chat")
        print(f"PERCENTUALE: {len(sampled)/total_available*100:.1f}%")
        print("="*50 + "\n")
    
    return sampled


def save_sample_manifest(sampled: List[Tuple[str, str, int]], output_path: Path):
    """Salva la lista delle chat campionate per tracking e riproducibilità."""
    manifest = {
        "total_sampled": len(sampled),
        "sampling_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42,
        "chats": [
            {
                "group": group,
                "chat_id": chat_id,
                "message_count": msg_count
            }
            for group, chat_id, msg_count in sampled
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Sample manifest salvato: {output_path}\n")


def load_sample_manifest(manifest_path: Path) -> List[Tuple[str, str, int]]:
    """Carica un manifest di campionamento esistente."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    sampled = [
        (chat['group'], chat['chat_id'], chat['message_count'])
        for chat in manifest['chats']
    ]
    
    print(f"✓ Caricato sample esistente: {len(sampled)} chat")
    print(f"  Data campionamento: {manifest['sampling_date']}\n")
    
    return sampled


def filter_db_by_sample(
    messages_db: Dict,
    sampled: List[Tuple[str, str, int]]
) -> Dict:
    """
    Filtra messages_db per includere solo le chat campionate.
    
    Returns:
        Nuovo dictionary con solo le chat campionate
    """
    filtered_db = {}
    
    # Crea set per lookup veloce
    sampled_set = {(group, chat_id) for group, chat_id, _ in sampled}
    
    for group, chats in messages_db.items():
        filtered_db[group] = {}
        for chat_id, chat_data in chats.items():
            if (group, chat_id) in sampled_set:
                filtered_db[group][chat_id] = chat_data
    
    # Rimuovi gruppi vuoti
    filtered_db = {g: c for g, c in filtered_db.items() if c}
    
    return filtered_db
