# Bioattend-raspberry
Script Python Raspberry Pi - Systeme de reconnaissance

## Initialiser une nouvelle machine (une seule commande)

Lancer le bootstrap automatique:

```bash
bash scripts/bootstrap_env.sh
```

Ce script:
- cree (ou reutilise) `venv`
- installe les paquets systeme necessaires
- installe automatiquement les dependances Python adaptees (`requirements.txt` sur Raspberry, `requirements-dev.txt` ailleurs)
- evite de reinstaller si les requirements n'ont pas change

Options utiles:

```bash
# forcer une reinstall complete des dependances
bash scripts/bootstrap_env.sh --force

# forcer mode production (Raspberry)
bash scripts/bootstrap_env.sh --prod

# forcer mode developpement
bash scripts/bootstrap_env.sh --dev
```
