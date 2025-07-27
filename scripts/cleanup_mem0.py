#!/usr/bin/env python3
"""Simple Mem0 cleanup script"""
import os
import requests
import json

def cleanup_mem0():
    """Clean up Mem0 memories using API calls."""
    api_key = os.getenv('MEM0_API_KEY')
    if not api_key:
        print("No MEM0_API_KEY found in environment")
        return
    
    base_url = "https://api.mem0.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    user_id = "athena_agent"
    
    try:
        # Get all memories
        response = requests.get(
            f"{base_url}/memories/?user_id={user_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            memories = response.json().get('memories', [])
            print(f"Found {len(memories)} memories to delete")
            
            # Delete each memory
            for memory in memories:
                memory_id = memory.get('id')
                if memory_id:
                    delete_response = requests.delete(
                        f"{base_url}/memories/{memory_id}/?user_id={user_id}",
                        headers=headers
                    )
                    if delete_response.status_code == 200:
                        print(f"Deleted memory: {memory_id}")
                    else:
                        print(f"Failed to delete memory {memory_id}: {delete_response.status_code}")
            
            print("✅ Mem0 cleanup completed")
        else:
            print(f"Failed to get memories: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error during Mem0 cleanup: {e}")

if __name__ == "__main__":
    cleanup_mem0()