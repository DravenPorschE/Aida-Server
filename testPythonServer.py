from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    # This gets the JSON body sent from your app/script
    data = request.json  
    
    # Extract the "text" field (whatever you sent as {"text": "..."} )
    user_text = data.get("text", "")

    # Prints the sented data
    print(user_text)
    
    # Do something with it (pass to your AI, etc.)
    response_text = f"AI response to: {user_text}"

    # Send result back as JSON
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)