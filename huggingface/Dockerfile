# Hugging Face Space (sdk: docker) for Fooocus2026.
# Pulls the image built by the GitHub CI (.github/workflows/build_container.yml):
# `edge` = main branch; use a version tag (2026.6.0) to pin the Space to a release.
FROM ghcr.io/mikecastrodemaria/fooocus2026:edge

# Spaces route traffic to app_port (README.md); the image listens on 7865 by default.
ENV CMDARGS="--listen --port 7860"

# Models and outputs live in DATADIR (/content/data in the image, ephemeral: the models
# are downloaded again at each cold start). With the Space's persistent storage enabled
# (Settings > Storage), set these three variables in Settings > Variables instead of here:
#   DATADIR=/data
#   config_path=/data/config.txt
#   config_example_path=/data/config_modification_tutorial.txt
# /data only exists when that storage is enabled; pointing DATADIR at it without it
# would fail at start (a non-root user cannot create /data).

EXPOSE 7860
