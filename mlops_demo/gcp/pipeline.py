import re
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output


def compile_pipeline(image_uri: str, output: Path) -> None:
    digest_uri = r"[a-z0-9-]+-docker\.pkg\.dev/[^/:@\s]+/[^/:@\s]+/[^:@\s]+@sha256:[a-f0-9]{64}"
    if not re.fullmatch(digest_uri, image_uri):
        raise ValueError("Use a regional Artifact Registry image URI pinned by SHA-256 digest.")
    if output.suffix not in {".yaml", ".yml"}:
        raise ValueError("The compiled pipeline path must have a .yaml or .yml extension.")

    @dsl.component(base_image=image_uri, install_kfp_package=False)
    def prepare_data(source_uri: str, seed: int, prepared: Output[Dataset]):
        from mlops_demo.gcp.steps import prepare_dataset

        prepared.metadata.update(prepare_dataset(source_uri, prepared.uri, seed))

    @dsl.component(base_image=image_uri, install_kfp_package=False)
    def train_model(prepared: Input[Dataset], candidate: Output[Model]):
        from mlops_demo.gcp.steps import train_candidate

        train_candidate(prepared.uri, candidate.uri)

    @dsl.component(base_image=image_uri, install_kfp_package=False)
    def evaluate_model(
        prepared: Input[Dataset],
        candidate: Input[Model],
        max_mae: float,
        require_approval: bool,
        report: Output[Artifact],
        metrics: Output[Metrics],
    ) -> bool:
        from mlops_demo.gcp.steps import evaluate_candidate

        assessment = evaluate_candidate(prepared.uri, candidate.uri, report.uri, max_mae)
        metrics.log_metric("candidate_mae_minutes", assessment.mae)
        metrics.log_metric("baseline_mae_minutes", assessment.baseline_mae)
        metrics.log_metric("mae_limit_minutes", assessment.max_mae)
        metrics.log_metric("holdout_rows", assessment.holdout_rows)
        report.metadata["passed"] = assessment.passed
        if require_approval and not assessment.passed:
            raise ValueError(
                f"Candidate MAE {assessment.mae:.3f} min did not pass the quality gate "
                f"(baseline {assessment.baseline_mae:.3f}, limit {assessment.max_mae:.3f})."
            )
        return assessment.passed

    @dsl.component(base_image=image_uri, install_kfp_package=False)
    def register_model(
        candidate: Input[Model],
        report: Input[Artifact],
        project: str,
        region: str,
        model_id: str,
        serving_image: str,
        registered: Output[Model],
    ):
        from mlops_demo.gcp.steps import register_approved_model

        registered.metadata.update(
            register_approved_model(
                candidate.uri,
                report.uri,
                registered.uri,
                serving_image,
                project=project,
                region=region,
                model_id=model_id,
            )
        )

    @dsl.pipeline(name="delivery-time-continuous-training")
    def continuous_training(
        source_uri: str,
        project: str,
        region: str,
        model_id: str,
        max_mae: float,
        seed: int = 42,
        require_approval: bool = False,
    ):
        prepared = prepare_data(source_uri=source_uri, seed=seed).set_caching_options(False)
        candidate = train_model(prepared=prepared.outputs["prepared"]).set_caching_options(False)
        evaluation = evaluate_model(
            prepared=prepared.outputs["prepared"],
            candidate=candidate.outputs["candidate"],
            max_mae=max_mae,
            require_approval=require_approval,
        ).set_caching_options(False)
        with dsl.If(evaluation.outputs["Output"] == True, name="quality-gate-passed"):
            register_model(
                candidate=candidate.outputs["candidate"],
                report=evaluation.outputs["report"],
                project=project,
                region=region,
                model_id=model_id,
                serving_image=image_uri,
            ).set_caching_options(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(continuous_training, package_path=str(output))
