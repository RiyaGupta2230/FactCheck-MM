from src.milestone4.app import app, load_models

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting FactCheck-MM application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
