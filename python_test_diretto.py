import requests



headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "phi4-mini",
    "messages": [{"role": "user", "content": "Ciao"}],
    "max_tokens": 10
}

print(f"Tentativo connessione a: {URL}")
print(f"Uso chiave che inizia con: {API_KEY[:10]}...")

try:
    response = requests.post(URL, headers=headers, json=data, verify=False) # verify=False ignora problemi di certificato SSL
    print(f"Codice risposta: {response.status_code}")
    print("Contenuto risposta:", response.text)
except Exception as e:
    print(f"Errore: {e}")
