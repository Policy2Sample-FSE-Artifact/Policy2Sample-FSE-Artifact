"""Top-to-bottom Policy2Sample orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from semantic_ir import ContextSnapshot, OnlineExactProjectionBackend

from .compiler import (
    CompilationResult,
    CompilationStatus,
    PolicyCompiler,
    direction_task_contract,
    positive_distance_task_contract,
    speed_cap_contract,
)
from .dsl import validate_dsl_document
from .realizer import RealizationResult, SamplingRealizer


@dataclass(frozen=True)
class PipelineResult:
    compilation: CompilationResult
    realization: RealizationResult | None


class Policy2SamplePipeline:
    def __init__(self, compiler: PolicyCompiler | None = None) -> None:
        self.compiler = compiler or PolicyCompiler()

    def _backend_for(self, compiled: CompilationResult) -> OnlineExactProjectionBackend:
        if compiled.sketch is None:
            raise ValueError("executable compilation has no sketch")
        if compiled.deployment is not None:
            deployment = compiled.deployment
            return OnlineExactProjectionBackend(
                deployment.schema,
                deployment.binding,
                deployment.base_contracts,
                deployment.context_schema,
            )
        deployment = self.compiler.deployment_for(compiled.sketch)
        contracts = list(deployment.base_contracts)
        if compiled.sketch.action == "move":
            contracts.extend(
                item
                for item in (
                    direction_task_contract(compiled.sketch.requested_direction),
                    positive_distance_task_contract() if compiled.sketch.require_motion else None,
                    speed_cap_contract(compiled.sketch.speed_cap),
                )
                if item is not None
            )
        return OnlineExactProjectionBackend(
            deployment.schema,
            deployment.binding,
            contracts,
            deployment.context_schema,
        )

    def realize(
        self,
        compiled: CompilationResult,
        snapshot: ContextSnapshot,
        *,
        current_context_version: Callable[[], str] | None = None,
    ) -> PipelineResult:
        if compiled.status != CompilationStatus.EXECUTABLE or compiled.dsl is None:
            return PipelineResult(compiled, None)
        validate_dsl_document(compiled.dsl)
        realizer = SamplingRealizer(compiled, self._backend_for(compiled))
        return PipelineResult(compiled, realizer.realize(snapshot, current_context_version=current_context_version))

    def run(self, policy_text: str, snapshot: ContextSnapshot) -> PipelineResult:
        return self.run_with_context_version(policy_text, snapshot, expected_context_version=None)

    def run_with_context_version(
        self,
        policy_text: str,
        snapshot: ContextSnapshot,
        expected_context_version: str | None,
        current_context_version: Callable[[], str] | None = None,
    ) -> PipelineResult:
        compiled = self.compiler.compile(policy_text)
        if expected_context_version is not None and snapshot.version != expected_context_version:
            stale = RealizationResult(
                status="STALE",
                action=None,
                action_text=None,
                domains=(),
                reason=f"context version {snapshot.version!r} does not match expected {expected_context_version!r}",
            )
            return PipelineResult(compiled, stale)
        return self.realize(compiled, snapshot, current_context_version=current_context_version)

    def run_llm_sketch(
        self,
        sketch: Mapping[str, Any],
        source_text: str,
        snapshot: ContextSnapshot,
    ) -> PipelineResult:
        return self.realize(self.compiler.compile_llm_sketch(sketch, source_text), snapshot)
