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
    Stratified sampling based on chat length (message count).
    
    Args:
        messages_db: Dictionary from messages.json {group: {chat_id: {dialogue: [...]}}}
        random_seed: Seed for reproducibility
        verbose: Print distribution statistics
    
    Returns:
        List of tuples (group, chat_id, message_count)
    """
    random.seed(random_seed)
    
    # Categorize chats by length
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
        print("DATASET DISTRIBUTION")
        print("="*50)
        for bin_name, chats in bins.items():
            print(f"{bin_name:>10}: {len(chats):>3} chats")
        total = sum(len(c) for c in bins.values())
        print(f"{'TOTAL':>10}: {total:>3} chats")
        print("="*50 + "\n")
    
    # Stratified Sampling Strategy
    sampled = []
    
    if verbose:
        print("STRATIFIED SAMPLING EXECUTION")
        print("-"*50)
    
    # SKIP chats <10 messages
    if verbose:
        print(f"Excluded chats <10 messages: {len(bins['<10'])}")
    
    # Sample 10-30: ~55%
    sample_10_30 = random.sample(bins['10-30'], k=min(50, len(bins['10-30'])))
    sampled.extend(sample_10_30)
    if verbose:
        print(f"Sampled 10-30: {len(sample_10_30)}/{len(bins['10-30'])}")
    
    # Sample 30-60: ~42%
    sample_30_60 = random.sample(bins['30-60'], k=min(25, len(bins['30-60'])))
    sampled.extend(sample_30_60)
    if verbose:
        print(f"Sampled 30-60: {len(sample_30_60)}/{len(bins['30-60'])}")
    
    # Sample 60-100: ~33%
    sample_60_100 = random.sample(bins['60-100'], k=min(10, len(bins['60-100'])))
    sampled.extend(sample_60_100)
    if verbose:
        print(f"Sampled 60-100: {len(sample_60_100)}/{len(bins['60-100'])}")
    
    # Sample 100-150: ~30%
    sample_100_150 = random.sample(bins['100-150'], k=min(3, len(bins['100-150'])))
    sampled.extend(sample_100_150)
    if verbose:
        print(f"Sampled 100-150: {len(sample_100_150)}/{len(bins['100-150'])}")
    
    # Sample >150: Top 2 longest + 1 random
    if len(bins['>150']) > 0:
        sorted_long = sorted(bins['>150'], key=lambda x: x[2], reverse=True)
        sample_150_plus = sorted_long[:2]
        if len(sorted_long) > 2:
            sample_150_plus.append(random.choice(sorted_long[2:]))
        sampled.extend(sample_150_plus)
        max_msg = max(x[2] for x in bins['>150'])
        if verbose:
            print(f"Sampled >150: {len(sample_150_plus)}/{len(bins['>150'])} (max len: {max_msg})")
    
    total_available = sum(len(c) for c in bins.values() if c != bins['<10'])
    
    if verbose:
        print("-"*50)
        print(f"TOTAL SAMPLED: {len(sampled)} chats")
        print(f"PERCENTAGE: {len(sampled)/total_available*100:.1f}% of eligible data")
        print("="*50 + "\n")
    
    return sampled


def save_sample_manifest(sampled: List[Tuple[str, str, int]], output_path: Path):
    """Save the sampled chat list for reproducibility."""
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
    
    print(f"Sample manifest saved: {output_path.name}\n")


def load_sample_manifest(manifest_path: Path) -> List[Tuple[str, str, int]]:
    """Load an existing sampling manifest."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    sampled = [
        (chat['group'], chat['chat_id'], chat['message_count'])
        for chat in manifest['chats']
    ]
    
    print(f"✓ Loaded existing sample: {len(sampled)} chats")
    print(f"  Date: {manifest['sampling_date']}\n")
    
    return sampled


def filter_db_by_sample(
    messages_db: Dict,
    sampled: List[Tuple[str, str, int]]
) -> Dict:
    """
    Filter messages_db to include only sampled chats.
    
    Returns:
        New dictionary containing only the sampled conversations.
    """
    filtered_db = {}
    
    # Create set for O(1) lookup
    sampled_set = {(group, chat_id) for group, chat_id, _ in sampled}
    
    for group, chats in messages_db.items():
        filtered_db[group] = {}
        for chat_id, chat_data in chats.items():
            if (group, chat_id) in sampled_set:
                filtered_db[group][chat_id] = chat_data
    
    # Remove empty groups
    filtered_db = {g: c for g, c in filtered_db.items() if c}
    
    return filtered_db
