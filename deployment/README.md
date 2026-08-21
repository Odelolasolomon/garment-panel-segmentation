# Deployment Guide

This directory contains a production-oriented deployment scaffold for the panel segmentation service. The model logic still lives in the assessment entrypoints (`predict.py` and `apply_fabric.py`); deployment wraps that code with `serve.py`.

## Local API

Install runtime dependencies:

```bash
pip install -r requirements-deploy.txt
```

Run the API locally:

```bash
MODEL_WEIGHTS=weights/best.pt uvicorn serve:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Predict a mask:

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@sample.jpg" \
  --output mask.png
```

Apply a swatch to a predicted mask:

```bash
curl -X POST http://localhost:8080/apply-fabric \
  -F "panel_name=front_body" \
  -F "image=@sample.jpg" \
  -F "mask=@mask.png" \
  -F "swatch=@swatch.png" \
  --output rendered.png
```

## Docker

The Docker image does not bake in `weights/*.pt`; model checkpoints should be mounted at runtime or copied from artifact storage by the platform.

```bash
docker build -t panel-seg:local .
docker compose up --build
```

## Kubernetes

Update the image reference in `deployment/k8s/kustomization.yaml`, make sure the `panel-seg-models` PVC contains `best.pt`, then apply:

```bash
kubectl apply -k deployment/k8s
```

Included manifests:

- `namespace.yaml`: isolated namespace.
- `configmap.yaml`: runtime configuration.
- `pvc.yaml`: model artifact mount point.
- `deployment.yaml`: CPU API deployment with probes and resource requests.
- `service.yaml`: internal ClusterIP service.
- `job-benchmark.yaml`: optional one-shot latency benchmark job.

For autoscaling, enable Metrics Server in the target cluster and create an HPA using the deployed `panel-seg-api` deployment as the target.

## Terraform

`deployment/terraform/aws` creates:

- ECR repository for the service image.
- Private versioned S3 bucket for model artifacts.
- IAM read policy for model artifacts.
- Optional IRSA role for an existing EKS service account when OIDC inputs are provided.

```bash
cd deployment/terraform/aws
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
```

## Kubeflow

The Kubeflow pipeline source is in `deployment/kubeflow/pipeline.py`. It compiles a validation pipeline that runs the CPU latency benchmark inside the service image.

```bash
pip install -r requirements-kubeflow.txt
python deployment/kubeflow/pipeline.py
```