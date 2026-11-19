"""OSFP4 quantization plugin for stock llm-compressor.

Importing this package registers the ``"osfp4"`` observer and makes
``OSFP4Modifier`` resolvable by name from YAML/string recipes.
"""

import sys

from llmcompressor.modifiers.factory import ModifierFactory

from llmcompressor_osfp4 import observers

# Diagonal imports its observer through this path inside LLM Compressor.
sys.modules.setdefault("llmcompressor.observers.osfp4", observers)

from llmcompressor_osfp4.modifiers import OSFP4Modifier

ModifierFactory.register("OSFP4Modifier", OSFP4Modifier)

__all__ = ["OSFP4Modifier"]
