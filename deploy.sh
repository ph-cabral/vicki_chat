#!/usr/bin/env bash
# Deploy de vicki_chat en el server (10.10.0.159).
# Lo usa el workflow de GitHub Actions (.github/workflows/deploy.yml)
# y tambien sirve a mano:  ssh server -> cd ~/projects/vicki_chat -> ./deploy.sh
#
# OJO: hace 'git reset --hard origin/main'. En el server NO se edita codigo,
# asi que cualquier cambio local ahi es basura y se descarta a proposito.
# El .env NO esta en git -> no se toca.
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE=docker-compose.yml
BRANCH=main

echo "==> [1/4] traigo main de GitHub"
git fetch --prune origin
git reset --hard "origin/$BRANCH"
echo "    commit: $(git log -1 --format='%h %s')"

echo "==> [2/4] build + up"
docker compose -f "$COMPOSE_FILE" up -d --build "$@"

echo "==> [3/4] limpio imagenes viejas"
docker image prune -f >/dev/null

echo "==> [4/4] estado"
docker compose -f "$COMPOSE_FILE" ps
