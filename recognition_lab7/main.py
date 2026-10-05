from src.web.app import app

def main():
    print("Сервер запущен. Откройте в браузере: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)

if __name__ == "__main__":
    main()