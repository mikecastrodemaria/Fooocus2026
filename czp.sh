#!/usr/bin/env bash
# custom-26 : protocole CLI de la famille crispz (voir fooocus_protocol.py et
# comics2crispz/docs/CLI_PROTOCOL.md). Toujours UNE ligne JSON sur stdout.
# Codes : 0 ok, 1 erreur d'execution, 2 spec invalide, 3 non supporte, 4 pas de route.
export CZP_CALLER_CWD="$(pwd)"
cd "$(dirname "$0")" || exit 4
PY="${FOOOCUS_PYTHON:-python3}"
exec $PY -s fooocus_protocol.py "$@"
