import json
import os

transcript_path = r"C:\Users\Admin\.gemini\antigravity-ide\brain\d5d5a10b-94ec-439d-914b-fa8ec071dc12\.system_generated\logs\transcript_full.jsonl"
out_path = r"C:\Users\Admin\.gemini\antigravity-ide\scratch\cafet\conversation_history.md"

def export_chat():
    lines = []
    lines.append("# CAFET Phase 7.3 and 7.4 Conversation History\n\n")
    
    if not os.path.exists(transcript_path):
        lines.append("*Transcript file not found.*")
    else:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    source = data.get("source", "")
                    content = data.get("content", "")
                    
                    if not content:
                        continue
                        
                    if source == "USER_EXPLICIT":
                        lines.append("## USER\n")
                        lines.append(f"{content}\n\n---\n")
                    elif source == "MODEL":
                        lines.append("## ASSISTANT\n")
                        lines.append(f"{content}\n\n---\n")
                        
                except json.JSONDecodeError:
                    continue

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    print(f"Exported to {out_path}")

if __name__ == "__main__":
    export_chat()
