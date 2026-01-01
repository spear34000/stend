def on_message(data):
    msg = data.get("msg")
    sender = data.get("sender")
    print(f"[AdvancedSkill] Recv: {sender} -> {msg}")

def on_stend_event(data):
    event = data.get("event")
    print(f"[AdvancedSkill] Alert! Stend Event: {event}")
    
    if event == "NICKNAME_CHANGE":
        print(f"  User {data.get('target_id')} changed name from {data.get('from')} to {data.get('to')}")
    elif event == "MESSAGE_DELETE":
        print(f"  Message deleted by {data.get('target_id')}: {data.get('deleted_content')}")
    elif event == "MESSAGE_HIDE":
        print(f"  Message hidden in chat {data.get('chat_id')} by user {data.get('user_id')}")
