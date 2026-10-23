import torch
import torch.nn as nn
import wikipediaapi
torch.__version__
device = "mps"

wiki = wikipediaapi.Wikipedia(
    user_agent="kickoff-football-bot/1.0 (aaravgarg1011@gmail.com)",  # Wikipedia requires this
    language="en",
)
clubs = ["FC Barcelona", "Manchester City F.C", "Liverpool F.C.", "Real Madrid CF"]
text = ""
for club in clubs:
    ...