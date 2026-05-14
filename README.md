# 📺 SNRT — Archives Sémantiques

Système de recherche sémantique dans les archives audiovisuelles de la SNRT.  
PFE Master SID · FSR Rabat · AABID Mohamed Amine · 2025-2026

---

## Structure du projet

```
snrt_search/
├── app.py                  ← Application Streamlit (point d'entrée)
├── config.py               ← Constantes et chemins
├── models.py               ← Chargement modèles (cached)
├── search.py               ← Logique de recherche sémantique
├── indexer.py              ← Pipeline transcription → index
├── utils.py                ← Utilitaires (timecode, texte, CSV…)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── pvc.yaml
├── data/
│   ├── snrt_index_v2.faiss
│   └── snrt_metadata_v2.json
├── model/                  ← Whisper LoRA (adapter_model.safetensors)
├── snrt_biencoder_v2/      ← Bi-encoder fine-tuné
├── videos/                 ← Vidéos SNRT
└── uploads/                ← Créé automatiquement
```

---

## Installation locale

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

> FFmpeg requis : https://ffmpeg.org/download.html — ajouter au PATH

---

## Docker

### Prérequis
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé

### Lancer avec docker-compose (recommandé)

```bash
# 1. Construire l'image
docker compose build

# 2. Lancer le conteneur
docker compose up -d

# 3. Accéder à l'application
# http://localhost:8501

# 4. Voir les logs
docker compose logs -f

# 5. Arrêter
docker compose down
```

### Commandes Docker utiles

```bash
# Construire l'image manuellement
docker build -t snrt-search:latest .

# Lancer le conteneur manuellement
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/snrt_biencoder_v2:/app/snrt_biencoder_v2 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/uploads:/app/uploads \
  --name snrt_archives \
  snrt-search:latest

# Vérifier que le conteneur tourne
docker ps

# Accéder au shell du conteneur
docker exec -it snrt_archives bash

# Supprimer le conteneur
docker rm -f snrt_archives
```

> **Note** : Les dossiers `data/`, `model/`, `snrt_biencoder_v2/`, `videos/` sont montés
> en tant que volumes — ils ne sont pas copiés dans l'image (trop lourds).

---

## Kubernetes

### Prérequis
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installé
- Un cluster Kubernetes disponible :
  - **Local** : [Minikube](https://minikube.sigs.k8s.io/docs/start/) ou [Docker Desktop K8s](https://docs.docker.com/desktop/kubernetes/)
  - **Cloud** : GKE, EKS, AKS

### Étape 1 — Démarrer Minikube (test local)

```bash
# Installer minikube (Windows via winget)
winget install Kubernetes.minikube

# Démarrer le cluster
minikube start --memory=6g --cpus=2

# Vérifier
kubectl get nodes
```

### Étape 2 — Pousser l'image dans un registry

```bash
# Option A : GitHub Container Registry (recommandé)
# Remplace YOUR_USERNAME par ton username GitHub
docker tag snrt-search:latest ghcr.io/YOUR_USERNAME/snrt-search:latest
docker push ghcr.io/YOUR_USERNAME/snrt-search:latest

# Option B : Docker Hub
docker tag snrt-search:latest YOUR_USERNAME/snrt-search:latest
docker push YOUR_USERNAME/snrt-search:latest
```

> Mets à jour `image:` dans `k8s/deployment.yaml` avec ton username.

### Étape 3 — Déployer sur Kubernetes

```bash
# 1. Créer les volumes persistants
kubectl apply -f k8s/pvc.yaml

# 2. Déployer l'application
kubectl apply -f k8s/deployment.yaml

# 3. Exposer le service
kubectl apply -f k8s/service.yaml

# 4. Vérifier le déploiement
kubectl get pods
kubectl get services

# 5. Accéder à l'app (Minikube)
minikube service snrt-search-service --url
```

### Commandes kubectl utiles

```bash
# Voir l'état des pods
kubectl get pods -w

# Voir les logs
kubectl logs -f deployment/snrt-search

# Décrire un pod (debug)
kubectl describe pod <pod-name>

# Shell dans un pod
kubectl exec -it <pod-name> -- bash

# Mettre à jour l'image (rolling update)
kubectl set image deployment/snrt-search snrt-search=ghcr.io/YOUR_USERNAME/snrt-search:v2

# Rollback
kubectl rollout undo deployment/snrt-search

# Supprimer tout
kubectl delete -f k8s/
```

---

## CI/CD — GitHub Actions

Le pipeline `.github/workflows/ci.yml` se déclenche automatiquement à chaque `git push` :

| Étape | Déclencheur | Action |
|-------|-------------|--------|
| **Lint** | Tout push | Vérifie le style du code (ruff) |
| **Tests** | Après lint | Lance `pytest tests/` |
| **Build & Push** | Push sur `main` | Construit l'image Docker et la publie sur GHCR |

### Configurer GitHub Actions

```bash
# 1. Initialiser git
git init
git add .
git commit -m "Initial commit"

# 2. Créer le repo sur GitHub, puis :
git remote add origin https://github.com/YOUR_USERNAME/snrt-search.git
git push -u origin main

# → Le pipeline CI/CD se lance automatiquement
```

---

## Fonctionnalités

| Onglet | Description |
|--------|-------------|
| 🔍 Rechercher | Recherche sémantique + re-ranking cross-encoder, filtres, lecteur vidéo au timecode |
| 📤 Indexer | Upload vidéo → FFmpeg → Whisper → Embeddings → FAISS |
| 📊 Statistiques | Métriques du corpus, distribution par langue, statut des modèles |
