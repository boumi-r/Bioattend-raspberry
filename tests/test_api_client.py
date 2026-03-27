# ============================================================
# tests/test_api_client.py
# Test de api_client.py avec une vraie image
# ============================================================
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
import api_client
import logging

# Activer les logs pour voir ce qui se passe
logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s [%(levelname)s] %(name)s : %(message)s"
)

print("=" * 55)
print("TEST API CLIENT — BioAttend Raspberry")
print("=" * 55)

# ── Test 1 : vérifier que le serveur est accessible ─────────
print("\n[TEST 1] Vérification serveur...")
ok = api_client.check_server()
print(f"  Résultat : {'OK' if ok else 'ECHEC'}")

# ── Test 2 : envoyer une image avec un visage ────────────────
print("\n[TEST 2] Envoi image avec visage...")

# Cherche une image de test dans le dossier courant
image_path = None
for name in ["test_face.jpg", "RAOUL.jpg", "face.jpg", "photo.jpg"]:
    if os.path.exists(name):
        image_path = name
        break

if image_path:
    print(f"  Image trouvée : {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    result = api_client.send_image(image_bytes)
    print(f"  success       : {result.get('success')}")
    print(f"  face_detected : {result.get('face_detected')}")
    if result.get("embedding"):
        print(f"  embedding     : {len(result['embedding'])} valeurs")
        print(f"  premiers      : {result['embedding'][:3]}")
    if result.get("error"):
        print(f"  erreur        : {result.get('error')}")
else:
    print("  Aucune image trouvée — upload une photo .jpg dans le dossier")

# ── Test 3 : envoyer des bytes vides (test erreur) ───────────
print("\n[TEST 3] Envoi image invalide (bytes vides)...")
result = api_client.send_image(b"ceci_nest_pas_une_image")
print(f"  success : {result.get('success')}")
print(f"  erreur  : {result.get('error')}")

print("\n" + "=" * 55)
print("TESTS TERMINÉS")
print("=" * 55)
