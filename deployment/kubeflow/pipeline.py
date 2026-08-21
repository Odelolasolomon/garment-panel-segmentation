"""Kubeflow pipeline for validating a panel segmentation model artifact.

Compile with:
    python deployment/kubeflow/pipeline.py

The pipeline expects the container image to contain this repository code. Model
weights and sample images should be mounted or downloaded by your platform layer
before the component starts.
"""
from pathlib import Path

from kfp import compiler, dsl


@dsl.container_component
def latency_benchmark(
    image_uri: str,
    weights_path: str,
    sample_image_path: str,
    runs: int,
):
    return dsl.ContainerSpec(
        image=image_uri,
        command=["python", "benchmark_latency.py"],
        args=[
            "--weights",
            weights_path,
            "--image",
            sample_image_path,
            "--runs",
            runs,
        ],
    )


@dsl.pipeline(
    name="panel-seg-validation",
    description="Run CPU latency validation for a trained garment panel segmentation checkpoint.",
)
def panel_seg_validation_pipeline(
    image_uri: str = "ghcr.io/your-org/panel-seg:latest",
    weights_path: str = "/app/weights/best.pt",
    sample_image_path: str = "/app/samples/sample.jpg",
    runs: int = 50,
):
    latency_benchmark(
        image_uri=image_uri,
        weights_path=weights_path,
        sample_image_path=sample_image_path,
        runs=runs,
    )


if __name__ == "__main__":
    output_path = Path(__file__).with_name("panel_seg_validation_pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=panel_seg_validation_pipeline,
        package_path=str(output_path),
    )
    print(f"Wrote {output_path}")