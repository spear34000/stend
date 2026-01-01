def on_message(chat, reply):
    msg = chat['message']['content']
    sender = chat['sender']['name']
    room = chat['room']['name']
    
    print(f"[{room}] {sender}: {msg}")
    
    if msg == "/ping":
        reply("Stend Platform Pong!")
    
    if msg == "/info":
        reply("I am running on Stend (Iris Zero) Platform.")
