"""MatEvolve: agent-based evolutionary design of crystal structures over motif profiles.

Package map (manuscript section in brackets):

* :mod:`matevolve.llm`          -- language-model clients (Claude Fable 5 at T = 0.5 in the paper)
* :mod:`matevolve.profile`      -- the motif profile text representation            [Sec. 2.1, SI S1]
* :mod:`matevolve.agents`       -- CIF2Text, Text2CIF, EvoOperator, Coverage         [Methods 2-3, SI S1-S3]
* :mod:`matevolve.translate`    -- Structure Translate / Motif Translate loops        [Methods 2-3, Alg. 2-3]
* :mod:`matevolve.clustering`   -- agentic consensus clustering of motifs            [Methods 4, Alg. 4]
* :mod:`matevolve.ers`          -- evidence-based recommender (Dempster-Shafer)      [Methods 5]
* :mod:`matevolve.validation`   -- VASP inputs, convex hull, magnetization           [Methods 6, SI S8]
* :mod:`matevolve.evaluation`   -- benchmark rates and reconstruction fidelity       [Sec. 2.2, SI S4]
* :mod:`matevolve.loop`         -- the generation loop                               [Methods 1, Alg. 1]
"""

__version__ = "1.0.0"

from .llm import AnthropicClient, LLMClient, LLMConfig, OpenAICompatibleClient, ScriptedClient  # noqa: F401
from .profile import DesignConcept, Motif, MotifProfile  # noqa: F401
