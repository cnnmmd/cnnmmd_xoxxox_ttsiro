#!/bin/bash

pthtop="$(cd "$(dirname "${0}")/../../../.." && pwd)"
source "${pthtop}"/manage/lib/params.sh
source "${pthtop}"/manage/lib/shared.sh
source "${pthcrr}"/params.sh

pthapp="${pthsrc}"/appiro
pthhgf="${pthapp}/hgf"
pthprm="${pthapp}/prm"
pthprj="${pthapp}/Irodori-TTS"
pthlib="${pthsrc}/export/lib/xoxxox"
cntapp='/opt/appiro'
cntprj="${cntapp}/Irodori-TTS"

addimg ${imgtgt} "${cnfimg}" "${pthdoc}"
test -d "${pthapp}" || mkdir "${pthapp}"
test -d "${pthhgf}" || mkdir "${pthhgf}"
test -d "${pthprm}" || mkdir "${pthprm}"
if cd "${pthapp}"
then
  test -d "${pthprj}" || git clone --depth 1 https://github.com/Aratako/Irodori-TTS.git
fi
test -d "${pthlib}/irodori_tts" || cp -r "${pthprj}/irodori_tts" "${pthlib}"

docker run -v "${pthapp}":"${cntapp}" --name ${cnttgt} ${imgtgt} /exp/runcfg.sh "${cntprj}" && \
docker commit ${cnttgt} ${imgtgt} && \
docker stop ${cnttgt} && \
docker rm ${cnttgt}
