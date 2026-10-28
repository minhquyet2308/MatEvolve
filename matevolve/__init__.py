"""MatEvolve: agent-based evolutionary design of crystal structures over motif profiles.

Package map:

* :mod:`matevolve.llm`          -- language-model clients (Claude Fable 5 at T = 0.5 in the reported runs)
* :mod:`matevolve.profile`      -- the motif profile text representation
* :mod:`matevolve.agents`       -- CIF2Text, Text2CIF, EvoOperator, Coverage
* :mod:`matevolve.translate`    -- Structure Translate / Motif Translate loops
* :mod:`matevolve.ers`          -- evidence-based recommender (Dempster-Shafer)
* :mod:`matevolve.database`     -- the structure database (CSV + CIF + profile per structure)
* :mod:`matevolve.dft_io`       -- hand-off of candidates to the external DFT tool
* :mod:`matevolve.loop`         -- the generation loop
"""

__version__ = "1.0.0"

from .llm import AnthropicClient, LLMClient, LLMConfig, OpenAICompatibleClient, ScriptedClient  # noqa: F401
from .profile import DesignConcept, Motif, MotifProfile  # noqa: F401
