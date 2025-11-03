import requests
import os

def test_ollama_connection():
    """Test de connectivité avec Ollama"""
    
    base_urls = [
        "http://ollama:11434",
        "http://ollama:11434/api",
        os.getenv("OLLAMA_API_BASE", ""),
    ]
    
    print("=" * 60)
    print("TEST DE CONNECTIVITÉ OLLAMA")
    print("=" * 60)
    
    for url in base_urls:
        if not url:
            continue
            
        print(f"\n🔍 Test de l'URL : {url}")
        
        # Test 1: Endpoint root
        try:
            response = requests.get(f"{url}", timeout=5)
            print(f"  ✅ GET {url} -> Status {response.status_code}")
        except Exception as e:
            print(f"  ❌ GET {url} -> Erreur: {e}")
        
        # Test 2: Endpoint /api/tags
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            print(f"  ✅ GET {url}/api/tags -> Status {response.status_code}")
            if response.status_code == 200:
                print(f"     Modèles disponibles: {response.json()}")
        except Exception as e:
            print(f"  ❌ GET {url}/api/tags -> Erreur: {e}")
        
        # Test 3: Endpoint /api/embed
        try:
            payload = {
                "model": "all-minilm:latest",
                "input": "test"
            }
            response = requests.post(f"{url}/api/embed", json=payload, timeout=10)
            print(f"  ✅ POST {url}/api/embed -> Status {response.status_code}")
        except Exception as e:
            print(f"  ❌ POST {url}/api/embed -> Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("VARIABLES D'ENVIRONNEMENT")
    print("=" * 60)
    print(f"OLLAMA_API_BASE = {os.getenv('OLLAMA_API_BASE', 'NON DÉFINIE')}")
    print(f"PYTHONPATH = {os.getenv('PYTHONPATH', 'NON DÉFINIE')}")
    print("=" * 60)

if __name__ == "__main__":
    test_ollama_connection()