"""Optional XGrammar structural layer.

The semantic layer does not depend on XGrammar being installed.  When it is
available, this wrapper compiles the static JSON structure and exposes the
matcher hooks needed by an engine-level mask intersection.
"""

from __future__ import annotations

from typing import Any, Mapping

from .structure import structural_schema_for_dsl


class XGrammarUnavailable(RuntimeError):
    pass


class XGrammarStructuralLayer:
    def __init__(self, tokenizer_info: Any) -> None:
        try:
            import xgrammar as xgr  # type: ignore
        except ImportError as exc:
            raise XGrammarUnavailable("install xgrammar to enable the structural layer") from exc
        self.xgr = xgr
        self.tokenizer_info = tokenizer_info
        self.compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)

    def compile_json_schema(self, schema: Mapping[str, object]) -> Any:
        return self.compiler.compile_json_schema(__import__("json").dumps(schema, ensure_ascii=False, sort_keys=True))

    def compile_dsl(self, dsl_document: Mapping[str, object]) -> Any:
        """Compile only the DSL's static structure; dynamic bounds stay external."""
        return self.compile_json_schema(structural_schema_for_dsl(dsl_document))

    def matcher(self, compiled_grammar: Any) -> Any:
        return self.xgr.GrammarMatcher(compiled_grammar)

    def fill_next_token_bitmask(self, matcher: Any, bitmask: Any) -> None:
        matcher.fill_next_token_bitmask(bitmask)
