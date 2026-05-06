#!/bin/bash

cntprj="${1}"
pthloc='/root/.local'

cd "${cntprj}" && \
${pthloc}/bin/uv sync --upgrade-package torchcodec --upgrade-package torch --upgrade-package torchaudio && \
${pthloc}/bin/uv add aiohttp
