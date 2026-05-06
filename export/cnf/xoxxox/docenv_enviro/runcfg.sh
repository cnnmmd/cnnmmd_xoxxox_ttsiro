#!/bin/bash

cntprj="${1}"
pthloc='/root/.local'

ARC=$(uname -m) && \
cd "${cntprj}" && \
if test "${ARC}" = 'x86_64'; \
then \
  cp pyproject.toml pyproject_old.toml && \
  sed -i \
    -e 's/index = "pytorch-cu128"/index = "pytorch-cpu"/g' \
    -e 's/name = "pytorch-cu128"/name = "pytorch-cpu"/g' \
    -e 's|url = "https://download.pytorch.org/whl/cu128"|url = "https://download.pytorch.org/whl/cpu"|g' \
    pyproject.toml
fi && \
${pthloc}/bin/uv lock --upgrade-package torch --upgrade-package torchaudio && \
${pthloc}/bin/uv sync && \
${pthloc}/bin/uv add aiohttp
