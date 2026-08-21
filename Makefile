.PHONY: install test serve docker-build docker-run k8s-apply terraform-fmt

install:
	python -m pip install -r requirements.txt

test:
	pytest tests/ -v

serve:
	uvicorn serve:app --host 0.0.0.0 --port 8080

docker-build:
	docker build -t panel-seg:local .

docker-run:
	docker compose up --build

k8s-apply:
	kubectl apply -k deployment/k8s

terraform-fmt:
	terraform -chdir=deployment/terraform/aws fmt -recursive