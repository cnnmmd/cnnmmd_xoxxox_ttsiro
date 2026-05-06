from xoxxox.shared import Custom
from xoxxox.libiro import init_tts, infer_tts

#---------------------------------------------------------------------------

class TtsPrc():

  def __init__(self, config="xoxxox/config_ttsiro_000", **dicprm):
    diccnf = Custom.update(config, dicprm)

  def status(self, config="xoxxox/config_ttsiro_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    keyspk = diccnf["keyspk"]
    option = diccnf["option"]
    cfgspk = diccnf["cfgspk"]
    cfgcap = diccnf["cfgcap"]
    cfgtxt = diccnf["cfgtxt"]
    if keyspk == "":
      pthwav = ""
    else:
      pthwav = "/opt/appiro/prm/" + keyspk + ".wav"
    self.objtts = init_tts(
      hf_checkpoint="Aratako/Irodori-TTS-500M-v2",
      ref_wav=pthwav,
      caption=option,
      cfg_scale_speaker=cfgspk,
      cfg_scale_caption=cfgcap,
      cfg_scale_text=cfgtxt,
      tmp_dir="/dev/shm",
    )

  async def infere(self, txtreq):
    datwav = await infer_tts(self.objtts, txtreq)
    return datwav
